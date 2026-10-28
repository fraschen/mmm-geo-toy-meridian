"""Phase 8: calibration with a HYPOTHETICAL lift test, before vs after.

- Convergence of the calibrated refit (same gate as the production model).
- ROI and P(ROI > 1) per channel, before (R1) and after (calibrated).
- The calibrated Channel3 prior vs the test result (CalibrationBuilder adjustments).
- What each model implies for Channel3's ROI in the test geos and weeks vs the test.
- Geo-choice check: implied ROI (uncalibrated model) and the calibrated prior for 20
  alternative population-stratified 10-geo sets.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "model_build"))

from common import (CHANNELS, GEO, OUTPUTS, ROOT, SEED, TIME, build_input_data,  # noqa: E402
                    load_model, load_raw, write_json)
from make_hypothetical_lift_test import N_TEST_GEOS, stratified_geos  # noqa: E402
from check_convergence import convergence  # noqa: E402

CI = (0.03, 0.97)
N_ALT = 20
OUT = OUTPUTS / "calibration"


def summarize(draws):
  draws = np.asarray(draws).ravel()
  return {"mean": float(draws.mean()), "ci_lo": float(np.quantile(draws, CI[0])),
          "ci_hi": float(np.quantile(draws, CI[1])), "p_roi_gt_1": float((draws > 1).mean())}


def roi(mmm, **kw):
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  return np.asarray(an.roi(use_kpi=False, **kw)).reshape(-1, len(CHANNELS))


def calibrated_prior_for(test_spend, test):
  """Calibrated Channel3 prior (mean, sd) for a given experiment spend."""
  import jax
  import build_model as bm
  from meridian.model.calibration import prior_builder
  tfd = bm.backend.tfd
  default = tfd.LogNormal(np.float64(0.2), np.float64(0.9), name="roi_m")
  b = prior_builder.CalibrationBuilder(build_input_data(), custom_prior=default,
                                       baseline_prior=default, max_lag=4)
  b = b.with_incrementality_experiment_result(
      test["channel"], point_estimates=[test["point_estimate_roi"]],
      standard_errors=[test["standard_error"]], experiment_kpi_types=[test["kpi_type"]],
      experiment_total_spends=[test_spend], experiment_start_dates=[test["start_date"]],
      experiment_end_dates=[test["end_date"]])
  x = np.asarray(b.build().roi_m.sample(20000, seed=jax.random.PRNGKey(0)))[:, CHANNELS.index(test["channel"])]
  return {"mean": float(x.mean()), "sd": float(x.std()), "p_roi_gt_1": float((x > 1).mean())}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  test = pd.read_csv(ROOT / "data" / "lift_tests" / "lift_tests.csv").iloc[0].to_dict()
  ch = test["channel"]
  ci = CHANNELS.index(ch)
  test_geos = test["test_geos"].split(";")
  raw = load_raw()
  weeks = sorted(str(t.date()) for t in raw[TIME].unique()
                 if test["start_date"] <= str(t.date()) <= test["end_date"])

  before, after = load_model(), load_model("calibrated")
  conv = convergence(after.inference_data)
  result = {"label": "HYPOTHETICAL lift test", "test": test,
            "after_converged": conv["converged"],
            "after_convergence": {k: conv[k] for k in ("max_rhat", "min_ess_bulk",
                                                       "min_ess_tail", "divergences", "passed")}}

  prior_after = np.asarray(after.inference_data.prior["roi_m"]).reshape(-1, len(CHANNELS))
  prior_before = np.asarray(before.inference_data.prior["roi_m"]).reshape(-1, len(CHANNELS))
  result["prior_channel"] = {"before": summarize(prior_before[:, ci]),
                             "after_calibrated": summarize(prior_after[:, ci]),
                             "after_sd": float(prior_after[:, ci].std())}

  # Implied ROI for the test geos and weeks (uncalibrated model is always valid).
  implied_before = roi(before, selected_geos=test_geos, selected_times=weeks)[:, ci]
  result["implied_test_roi_before"] = summarize(implied_before)

  rows = []
  roi_b = roi(before)
  if conv["converged"]:
    roi_a = roi(after)
    result["implied_test_roi_after"] = summarize(
        roi(after, selected_geos=test_geos, selected_times=weeks)[:, ci])
    for i, c in enumerate(CHANNELS):
      b, a = summarize(roi_b[:, i]), summarize(roi_a[:, i])
      rows.append({"channel": c, **{f"before_{k}": v for k, v in b.items()},
                   **{f"after_{k}": v for k, v in a.items()}})
  else:
    # A non-converged fit records only its convergence diagnostics.
    for i, c in enumerate(CHANNELS):
      rows.append({"channel": c, **{f"before_{k}": v for k, v in summarize(roi_b[:, i]).items()}})
  pd.DataFrame(rows).to_csv(OUT / "roi_before_after.csv", index=False)
  result["roi_before_after"] = rows

  # Geo-choice check.
  alt = []
  pops = raw.groupby(GEO)["population"].first()
  for k in range(N_ALT):
    geos = stratified_geos(raw, N_TEST_GEOS, SEED + 1000 + k)
    window = (raw[TIME] >= test["start_date"]) & (raw[TIME] <= test["end_date"]) & raw[GEO].isin(geos)
    spend = float(raw.loc[window, f"{ch}_spend"].sum())
    imp = roi(before, selected_geos=geos, selected_times=weeks)[:, ci]
    alt.append({"set": k, "geos": ";".join(geos), "population_share": float(pops[geos].sum() / pops.sum()),
                "test_spend": spend, "implied_roi_mean": float(imp.mean()),
                "implied_p_roi_gt_1": float((imp > 1).mean()),
                **{f"calibrated_prior_{kk}": v for kk, v in calibrated_prior_for(spend, test).items()}})
  alt_df = pd.DataFrame(alt)
  alt_df.to_csv(OUT / "geo_choice_check.csv", index=False)
  chosen_prior = calibrated_prior_for(test["total_spend"], test)
  result["geo_choice_check"] = {
      "n_alternative_sets": N_ALT,
      "chosen_set": {"implied_roi_mean": result["implied_test_roi_before"]["mean"],
                     "calibrated_prior": chosen_prior,
                     "population_share": float(pops[test_geos].sum() / pops.sum())},
      "implied_roi_mean_range": [float(alt_df.implied_roi_mean.min()), float(alt_df.implied_roi_mean.max())],
      "implied_roi_mean_sd": float(alt_df.implied_roi_mean.std()),
      "chosen_implied_percentile": float((alt_df.implied_roi_mean < result["implied_test_roi_before"]["mean"]).mean()),
      "calibrated_prior_mean_range": [float(alt_df.calibrated_prior_mean.min()), float(alt_df.calibrated_prior_mean.max())],
      "calibrated_prior_sd_range": [float(alt_df.calibrated_prior_sd.min()), float(alt_df.calibrated_prior_sd.max())],
      "national_roi_before": summarize(roi_b[:, ci]),
  }
  write_json(result, OUT / "calibration.json")
  print({k: result[k] for k in ("after_converged", "after_convergence", "prior_channel",
                                "implied_test_roi_before")})
  print(result["geo_choice_check"])
  for r in rows:
    print(r)


if __name__ == "__main__":
  main()
