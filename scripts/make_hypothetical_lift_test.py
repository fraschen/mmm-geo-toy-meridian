"""Generates a HYPOTHETICAL lift test for Channel3 (Phase 8).

This is an invented experiment, not a real result. It is written in the file
format documented in data/lift_tests/README.md, so a real test can replace the row.

Design (fixed before looking at the model's Channel3 estimate):
  - Channel3, the largest channel with undetermined payback after the R1 fit.
  - 8 weeks, 2023-07-03 to 2023-08-27; 10 test geos drawn by population-stratified
    random sampling (one geo from each population decile pair, fixed seed).
  - Reported ROI 1.4 (revenue per unit spend) with standard error 0.25, against a
    zero-spend counterfactual (the MMM's estimand).
  - Spend = Channel3's actual spend in the test geos and weeks.
"""

import numpy as np
import pandas as pd

from common import GEO, POPULATION, ROOT, SEED, TIME, load_raw

OUT = ROOT / "data" / "lift_tests"
START, END = "2023-07-03", "2023-08-27"
N_TEST_GEOS = 10


def stratified_geos(df, n, seed):
  pop = df.groupby(GEO)[POPULATION].first().sort_values()
  strata = np.array_split(pop.index.to_numpy(), n)
  rng = np.random.default_rng(seed)
  return sorted(str(rng.choice(s)) for s in strata)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  df = load_raw()
  test_geos = stratified_geos(df, N_TEST_GEOS, SEED)
  control_geos = sorted(set(df[GEO].unique()) - set(test_geos))
  window = (df[TIME] >= START) & (df[TIME] <= END) & df[GEO].isin(test_geos)
  spend = float(df.loc[window, "Channel3_spend"].sum())
  row = {
      "test_id": "HYPOTHETICAL-CH3-2023Q3",
      "hypothetical": True,
      "channel": "Channel3",
      "kpi_type": "revenue",
      "point_estimate_roi": 1.4,
      "standard_error": 0.25,
      "counterfactual": "zero_spend",
      "total_spend": round(spend, 2),
      "start_date": START,
      "end_date": END,
      "test_geos": ";".join(test_geos),
      "control_geos": ";".join(control_geos),
      "source": "HYPOTHETICAL: invented for demonstration by scripts/make_hypothetical_lift_test.py",
  }
  pd.DataFrame([row]).to_csv(OUT / "lift_tests.csv", index=False)
  print(f"Wrote {OUT / 'lift_tests.csv'} (HYPOTHETICAL; test geos {test_geos}, spend {spend:,.0f})")


if __name__ == "__main__":
  main()
