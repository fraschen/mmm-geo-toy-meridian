# Lift test results (calibration input)

> **The only row currently in `lift_tests.csv` is HYPOTHETICAL.** It was invented by
> `04_scenarios/make_hypothetical_lift_test.py` to demonstrate calibration. It is not
> an experiment result.

`lift_tests.csv` holds one row per experiment. Replace or add rows with real results.
The calibration step (`02_fit/build_model.py --calibrated`) passes every row to Meridian's
`CalibrationBuilder.with_incrementality_experiment_result`.

| column | type | meaning |
|---|---|---|
| `test_id` | string | Unique identifier. |
| `hypothetical` | bool | `True` for invented rows. Reports label these HYPOTHETICAL. |
| `channel` | string | Meridian media channel name (`Channel0` ... `Channel4`). |
| `kpi_type` | `revenue` or `non-revenue` | Unit of the point estimate: ROI in revenue per unit spend, or incremental KPI per unit spend. |
| `point_estimate_roi` | float | Average incremental return per unit spend over the test. |
| `standard_error` | float | Standard error of the point estimate. |
| `counterfactual` | string | Must be `zero_spend`. Marginal-lift tests are an estimand mismatch (Meridian docs). |
| `total_spend` | float | Channel spend during the test, in the test geos. |
| `start_date`, `end_date` | `YYYY-MM-DD` | Test window (inclusive). |
| `test_geos`, `control_geos` | `;`-separated geo names | Treated and control geos. Used for the test-vs-model comparison. |
| `source` | string | Where the result comes from. |
