"""Phase 8: calibration with a HYPOTHETICAL lift test, before vs after.

- Convergence of the calibrated refit (same gate as the production model).
- ROI and P(ROI > 1) per channel, before (R1) and after (calibrated).
- The calibrated Channel3 prior vs the test result (CalibrationBuilder adjustments).
- What each model implies for Channel3's ROI in the test geos and weeks vs the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CHANNELS, OUTPUTS, ROOT, TIME, convergence, load_model, load_raw, summarize,
                    write_json)

OUT = OUTPUTS / "04_scenarios"


def roi_summary(draws):
  """Mean, 94% interval and P(ROI > 1) of ROI draws."""
  return {**summarize(draws), "p_roi_gt_1": float((np.asarray(draws) > 1).mean())}


def roi(mmm, **kw):
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  return np.asarray(an.roi(use_kpi=False, **kw)).reshape(-1, len(CHANNELS))


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
  result = {"label": "lift test", "test": test,
            "after_converged": conv["converged"],
            "after_convergence": {k: conv[k] for k in ("max_rhat", "min_ess_bulk",
                                                       "min_ess_tail", "divergences", "passed")}}

  prior_after = np.asarray(after.inference_data.prior["roi_m"]).reshape(-1, len(CHANNELS))
  prior_before = np.asarray(before.inference_data.prior["roi_m"]).reshape(-1, len(CHANNELS))
  result["prior_channel"] = {"before": roi_summary(prior_before[:, ci]),
                             "after_calibrated": roi_summary(prior_after[:, ci]),
                             "after_sd": float(prior_after[:, ci].std())}

  # Implied ROI for the test geos and weeks (uncalibrated model is always valid).
  implied_before = roi(before, selected_geos=test_geos, selected_times=weeks)[:, ci]
  result["implied_test_roi_before"] = roi_summary(implied_before)

  rows = []
  roi_b = roi(before)
  if conv["converged"]:
    roi_a = roi(after)
    result["implied_test_roi_after"] = roi_summary(
        roi(after, selected_geos=test_geos, selected_times=weeks)[:, ci])
    for i, c in enumerate(CHANNELS):
      b, a = roi_summary(roi_b[:, i]), roi_summary(roi_a[:, i])
      rows.append({"channel": c, **{f"before_{k}": v for k, v in b.items()},
                   **{f"after_{k}": v for k, v in a.items()}})
  else:
    # A non-converged fit records only its convergence diagnostics.
    for i, c in enumerate(CHANNELS):
      rows.append({"channel": c, **{f"before_{k}": v for k, v in roi_summary(roi_b[:, i]).items()}})
  pd.DataFrame(rows).to_csv(OUT / "roi_before_after.csv", index=False)
  result["roi_before_after"] = rows

  write_json(result, OUT / "calibration.json")
  print({k: result[k] for k in ("after_converged", "after_convergence", "prior_channel",
                                "implied_test_roi_before")})
  for r in rows:
    print(r)


if __name__ == "__main__":
  main()
