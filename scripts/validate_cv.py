"""Phase 4: 13-week holdout + rolling-origin CV for the selected spec.

Folds train on weeks 1-104, 1-117, 1-130, 1-143 and forecast the next 13 weeks;
the 1-143 fold is the final 13-week holdout. Each fold model is fitted on the
truncated data (model_build/build_model.py --train-weeks), so every data-derived
scaling (KPI mean/sd, media medians, control centring) comes from training weeks.

Forecasts use Meridian's own mean function (Analyzer.get_kpi_means) on inputs
scaled by the fold's transformers (DataTensorsBuilder.build_scaled_inputs), with
the knots=2 baseline extended along its line. They are conditional on the
observed media, controls, Promo and organic in the forecast weeks. Predictive
draws add Normal(0, sigma) on Meridian's transformed KPI scale, are mapped back
with the fold's KpiTransformer, and are converted to revenue with the observed
revenue_per_conversion.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from common import (GEO, OUTPUTS, RAW_CSV, SEED, TIME, load_model, load_model_config,
                    build_input_data, load_raw, write_json)
from check_convergence import convergence

_VAL = load_model_config()["validation"]
SPEC = load_model_config()["model"]["name"]
FOLDS = _VAL["train_weeks"]
HORIZON = _VAL["horizon"]
CI = (0.03, 0.97)  # 94% interval
OUT = OUTPUTS / "validation"


def fold_label(t0):
  return "holdout" if t0 == FOLDS[-1] else f"fold_{t0}"


def load_fold(t0):
  return load_model(train_weeks=t0)



def window_input(t0):
  df = pd.read_csv(RAW_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
  weeks = sorted(df[TIME].unique())[: t0 + HORIZON]
  return build_input_data(df[df[TIME].isin(weeks)])


def kpi_means(an, data, dist, n_times):
  """Analyzer._get_kpi_means with n_times_output passed to the media transform.

  Mirrors meridian.analysis.analyzer.Analyzer._get_kpi_means (google-meridian
  2.1.0), which does not forward n_times_output when the time axis differs from
  the fitted model's.
  """
  from meridian import backend
  tau_gt = backend.expand_dims(dist.tau_g, -1) + backend.expand_dims(dist.mu_t, -2)
  media, beta = an._get_transformed_media_and_beta(data_tensors=data, dist_tensors=dist,
                                                    n_times_output=n_times)
  result = tau_gt + backend.einsum("...gtm,...gm->...gt", media, beta)
  if data.controls is not None:
    result += backend.einsum("...gtc,...gc->...gt", data.controls, dist.gamma_gc)
  if data.non_media_treatments is not None:
    result += backend.einsum("...gtm,...gm->...gt", data.non_media_treatments, dist.gamma_gn)
  return np.asarray(result)


def forecast(mmm, t0):
  """Returns expected (mean) and predictive draws of revenue, (chain, draw, geo, time)."""
  from meridian import backend
  from meridian.analysis import tensors as T
  from meridian.analysis.analyzer import Analyzer

  ctx = mmm.model_context
  assert list(np.asarray(mmm.knot_info.knot_locations)) == [0, t0 - 1], "expects knots=2"
  win = window_input(t0)
  arr = lambda x: backend.to_tensor(np.asarray(x), dtype=backend.float_dtype)
  new = T.DataTensors(media=arr(win.media), media_spend=arr(win.media_spend),
                      organic_media=arr(win.organic_media),
                      non_media_treatments=arr(win.non_media_treatments),
                      controls=arr(win.controls), revenue_per_kpi=arr(win.revenue_per_kpi),
                      time=backend.to_tensor(np.asarray(win.time)))
  inp = T.DataTensorsBuilder(ctx).build_scaled_inputs(new_data=new,
                                                      include_non_paid_channels=True)
  data = dataclasses.replace(inp.tensors, time=None)

  post = mmm.inference_data.posterior
  kv = np.asarray(post["knot_values"])
  t = np.arange(t0 + HORIZON)
  mu_t = kv[..., :1] + (kv[..., 1:2] - kv[..., :1]) * t / (t0 - 1)
  names = [f.name for f in dataclasses.fields(T.DistributionTensors)]
  dist = {n: np.asarray(post[n]) for n in names if n in post}
  dist["mu_t"] = mu_t
  an = Analyzer(model_context=ctx, inference_data=mmm.inference_data)
  mean_scaled = kpi_means(an, data, T.DistributionTensors(**dist), t0 + HORIZON)

  # Check: on training weeks the forecast path reproduces Meridian's in-sample
  # expected outcome exactly (same transformers, same equations).
  ref = np.asarray(an.expected_outcome(use_kpi=True, aggregate_geos=False,
                                       aggregate_times=False,
                                       inverse_transform_outcome=False))
  err = float(np.abs(mean_scaled[..., :t0] - ref).max())
  assert err < 1e-8, f"forecast path does not reproduce in-sample means ({err})"

  sigma = np.asarray(post["sigma"])
  sigma = sigma[..., None, None] if sigma.ndim == 2 else sigma[..., None]
  rng = np.random.default_rng(SEED + t0)
  pred_scaled = mean_scaled + sigma * rng.standard_normal(mean_scaled.shape)
  inv = lambda x: np.asarray(ctx.kpi_transformer.inverse(arr(x)))
  rpk = np.asarray(win.revenue_per_kpi)
  return inv(mean_scaled) * rpk, inv(pred_scaled) * rpk, win


def score(obs, mean, pred, mask):
  """MAPE, coverage and bias over the time mask, at geo-week and national level."""
  lo, hi = np.quantile(pred, CI, axis=(0, 1))
  m = mean.mean((0, 1))
  o = obs
  g = {"mape": float(np.mean(np.abs(m[:, mask] / o[:, mask] - 1))),
       "coverage_94": float(np.mean((o[:, mask] >= lo[:, mask]) & (o[:, mask] <= hi[:, mask]))),
       "mean_signed_pct_error": float(np.mean(m[:, mask] / o[:, mask] - 1))}
  nat_pred = pred.sum(2)
  nlo, nhi = np.quantile(nat_pred, CI, axis=(0, 1))
  nm, no = mean.sum(2).mean((0, 1)), o.sum(0)
  n = {"mape": float(np.mean(np.abs(nm[mask] / no[mask] - 1))),
       "coverage_94": float(np.mean((no[mask] >= nlo[mask]) & (no[mask] <= nhi[mask]))),
       "mean_signed_pct_error": float(np.mean(nm[mask] / no[mask] - 1)),
       "share_weeks_over_predicted": float(np.mean(nm[mask] > no[mask]))}
  return g, n, (nm, nlo, nhi, no)


def fold_roi(mmm):
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  roi = np.asarray(an.roi(use_kpi=False))  # (chain, draw, channel)
  draws = roi.reshape(-1, roi.shape[-1])
  ch = list(np.asarray(mmm.input_data.media_channel))
  return {c: {"mean": float(draws[:, i].mean()), "ci_lo": float(np.quantile(draws[:, i], CI[0])),
              "ci_hi": float(np.quantile(draws[:, i], CI[1])),
              "p_roi_gt_1": float((draws[:, i] > 1).mean())} for i, c in enumerate(ch)}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  raw = load_raw()
  rows, roi_rows, nat_rows, folds = [], [], [], {}
  for t0 in FOLDS:
    mmm = load_fold(t0)
    conv = convergence(mmm.inference_data)
    label = fold_label(t0)
    conv_summary = {k: conv[k] for k in ("max_rhat", "min_ess_bulk", "min_ess_tail",
                                         "divergences", "passed")}
    if not conv["converged"]:
      # A non-converged fit records only its convergence diagnostics.
      folds[label] = {"label": label, "train_weeks": t0, "converged": False,
                      "convergence": conv_summary}
      rows.append({"fold": label, "train_weeks": f"1-{t0}",
                   "test_weeks": f"{t0 + 1}-{t0 + HORIZON}", "converged": False})
      print(f"{label}: NOT converged; diagnostics only: {conv_summary}")
      del mmm
      continue
    mean, pred, win = forecast(mmm, t0)
    weeks = sorted(raw[TIME].unique())[: t0 + HORIZON]
    obs = (raw[raw[TIME].isin(weeks)].pivot(index=GEO, columns=TIME, values="revenue")
           .loc[np.asarray(win.geo)].to_numpy())
    test = np.arange(t0 + HORIZON) >= t0
    g_in, n_in, _ = score(obs, mean, pred, ~test)
    g_out, n_out, (nm, nlo, nhi, no) = score(obs, mean, pred, test)
    rec = {"label": label, "train_weeks": t0, "test_weeks": [t0 + 1, t0 + HORIZON],
           "converged": True, "convergence": conv_summary,
           "geo_week": {"in_sample": g_in, "forecast": g_out,
                        "error_ratio": g_out["mape"] / g_in["mape"]},
           "national": {"in_sample": n_in, "forecast": n_out,
                        "error_ratio": n_out["mape"] / n_in["mape"]}}
    folds[label] = rec
    rows.append({"fold": label, "train_weeks": f"1-{t0}", "test_weeks": f"{t0 + 1}-{t0 + HORIZON}",
                 "converged": conv["converged"],
                 "geo_mape_in": g_in["mape"], "geo_mape_fc": g_out["mape"],
                 "geo_error_ratio": rec["geo_week"]["error_ratio"],
                 "geo_coverage_fc": g_out["coverage_94"],
                 "nat_mape_in": n_in["mape"], "nat_mape_fc": n_out["mape"],
                 "nat_error_ratio": rec["national"]["error_ratio"],
                 "nat_coverage_fc": n_out["coverage_94"],
                 "nat_bias_fc": n_out["mean_signed_pct_error"]})
    for i, w in enumerate(weeks):
      nat_rows.append({"fold": label, "time": str(pd.Timestamp(w).date()), "week": i + 1,
                       "forecast_period": bool(test[i]), "observed": no[i], "mean": nm[i],
                       "ci_lo": nlo[i], "ci_hi": nhi[i]})
    # ROI only for converged fold fits (never report a non-converged fit's ROI).
    if conv["converged"]:
      for c, v in fold_roi(mmm).items():
        roi_rows.append({"fold": label, "channel": c, **v})
    print(f"{label}: converged={conv['converged']} geo MAPE {g_out['mape']:.3f} "
          f"(ratio {rec['geo_week']['error_ratio']:.2f}, cov {g_out['coverage_94']:.2f}) | "
          f"national MAPE {n_out['mape']:.3f} (ratio {rec['national']['error_ratio']:.2f}, "
          f"cov {n_out['coverage_94']:.2f}, bias {n_out['mean_signed_pct_error']:+.3f})")
    del mmm

  pd.DataFrame(rows).to_csv(OUT / "cv_folds.csv", index=False)
  pd.DataFrame(nat_rows).to_csv(OUT / "cv_national_forecasts.csv", index=False)
  pd.DataFrame(roi_rows).to_csv(OUT / "cv_roi.csv", index=False)
  write_json({"spec": SPEC, "folds": folds, "horizon_weeks": HORIZON, "interval": CI,
              "method": __doc__.strip()}, OUT / "cv_metrics.json")
  print(f"Wrote {OUT}")


if __name__ == "__main__":
  main()
