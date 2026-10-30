"""Phase 4: 13-week holdout + rolling-origin CV for the production model.

Each fold model is fitted on weeks 1 to N + 13 with the last 13 weeks held out
(scripts/02_fit/build_model.py --train-weeks N, using Meridian's `holdout` in ModelSpec):
those weeks' revenue is left out of the fit, while their media, controls and Promo
are kept. Folds hold out weeks 105-117, 118-130, 131-143 and 144-156; the last is
the final 13-week holdout.

Accuracy on the training and held-out weeks comes from Meridian's
Analyzer.predictive_accuracy (MAPE, wMAPE and R2, per geo-week and national).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CI, OUTPUTS, TIME, convergence, load_model, load_model_config, load_raw, write_json

CONFIG = load_model_config()
FOLDS = CONFIG["validation"]["train_weeks"]
HORIZON = CONFIG["validation"]["horizon"]
OUT = OUTPUTS / "03_validate"


def fold_label(t0):
  return "holdout" if t0 == FOLDS[-1] else f"fold_{t0}"


def accuracy(an):
  """Meridian's MAPE, wMAPE and R2 on training ("in_sample") and held-out ("forecast") weeks."""
  acc = an.predictive_accuracy()["value"]
  out = {}
  for level, granularity in (("geo_week", "geo"), ("national", "national")):
    out[level] = {
        name: {"mape": float(acc.sel(metric="MAPE", geo_granularity=granularity, evaluation_set=split)),
               "wmape": float(acc.sel(metric="wMAPE", geo_granularity=granularity, evaluation_set=split)),
               "r2": float(acc.sel(metric="R_Squared", geo_granularity=granularity, evaluation_set=split))}
        for name, split in (("in_sample", "Train"), ("forecast", "Test"))}
    out[level]["error_ratio"] = out[level]["forecast"]["mape"] / out[level]["in_sample"]["mape"]
  return out


def national_series(an, raw, label, t0):
  """Observed vs model national weekly revenue, with the 94% interval of the model's mean."""
  draws = np.asarray(an.expected_outcome(use_kpi=False, aggregate_geos=True, aggregate_times=False))
  draws = draws.reshape(-1, draws.shape[-1])                  # (draw, week)
  observed = raw.groupby(TIME)["revenue"].sum().iloc[:t0 + HORIZON]
  return pd.DataFrame({
      "fold": label, "time": observed.index.date.astype(str), "week": np.arange(1, t0 + HORIZON + 1),
      "forecast_period": np.arange(t0 + HORIZON) >= t0, "observed": observed.to_numpy(),
      "mean": draws.mean(0), "ci_lo": np.quantile(draws, CI[0], 0), "ci_hi": np.quantile(draws, CI[1], 0)})


def fold_roi(an, channels):
  roi = np.asarray(an.roi(use_kpi=False))
  draws = roi.reshape(-1, roi.shape[-1])                      # (draw, channel)
  return {c: {"mean": float(draws[:, i].mean()), "ci_lo": float(np.quantile(draws[:, i], CI[0])),
              "ci_hi": float(np.quantile(draws[:, i], CI[1])),
              "p_roi_gt_1": float((draws[:, i] > 1).mean())} for i, c in enumerate(channels)}


def main():
  from meridian.analysis.analyzer import Analyzer
  OUT.mkdir(parents=True, exist_ok=True)
  raw = load_raw()
  roi_rows, national, folds = [], [], {}
  for t0 in FOLDS:
    mmm = load_model(train_weeks=t0)
    label = fold_label(t0)
    conv = convergence(mmm.inference_data)
    rec = {"label": label, "train_weeks": t0, "test_weeks": [t0 + 1, t0 + HORIZON],
           "converged": conv["converged"],
           "convergence": {k: conv[k] for k in ("max_rhat", "min_ess_bulk", "min_ess_tail",
                                                "divergences", "passed")}}
    folds[label] = rec
    if not conv["converged"]:
      # A non-converged fit records only its convergence diagnostics.
      print(f"{label}: NOT converged; diagnostics only: {rec['convergence']}")
      continue

    an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    rec.update(accuracy(an))
    nat = national_series(an, raw, label, t0)
    test = nat[nat.forecast_period]
    error = test["mean"] / test["observed"] - 1               # national weekly % error
    rec["national"]["forecast"]["mean_signed_pct_error"] = float(error.mean())
    rec["national"]["forecast"]["share_weeks_over_predicted"] = float((error > 0).mean())
    national.append(nat)
    for c, v in fold_roi(an, list(np.asarray(mmm.input_data.media_channel))).items():
      roi_rows.append({"fold": label, "channel": c, **v})
    print(f"{label}: geo MAPE {rec['geo_week']['forecast']['mape']:.3f} "
          f"(ratio {rec['geo_week']['error_ratio']:.2f}) | national MAPE "
          f"{rec['national']['forecast']['mape']:.3f} (ratio {rec['national']['error_ratio']:.2f}, "
          f"bias {rec['national']['forecast']['mean_signed_pct_error']:+.3f})")

  (pd.concat(national) if national else pd.DataFrame()).to_csv(OUT / "cv_national_forecasts.csv", index=False)
  pd.DataFrame(roi_rows).to_csv(OUT / "cv_roi.csv", index=False)
  write_json({"spec": CONFIG["model"]["name"], "folds": folds, "horizon_weeks": HORIZON, "interval": CI,
              "method": __doc__.strip()}, OUT / "cv_metrics.json")
  print(f"Wrote {OUT}")


if __name__ == "__main__":
  main()
