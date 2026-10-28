# Geo-level Bayesian MMM with Google Meridian

A reproducible Meridian (JAX backend) marketing mix model of Meridian's simulated geo
dataset: 40 geos x 156 weeks, 5 paid channels, revenue as the outcome.

## Verdict

**Not yet good enough for a client decision.** The selected model (R1: a
linear-trend baseline with strongly pooled media effects and a 4-week max lag) converges on the full
data, fits well in sample, and its intervals are well calibrated in sample. Three things stop it
from being client-ready:

- **Out-of-sample evidence is thin.** Only 1 of 4 rolling-origin CV
  fold fits converged. In that fold, national 94% interval coverage was 77%, below the 80% gate.
  ROI stability across folds could not be assessed.
- **No channel's payback is determined.** P(ROI > 1) ranges from 0.24 to
  0.81, and every 94% ROI interval spans 1.
- **Baseline and media are only weakly separated.** Total baseline and total media contribution
  correlate at -0.97 across posterior draws.

The candidate set and the convergence criteria were revised after earlier rounds failed to
converge (see [diagnostics](reports/diagnostics.md#how-the-model-was-chosen)).

Any number shared from this model should carry these caveats: national results only, wide
ROI intervals, thin out-of-sample validation, and budget gains whose credible intervals
include losses.

## Reproduce

```bash
make all      # every phase: data, EDA, fits, selection, CV, health checks, budget, calibration, reports
make gates    # automated gates; exits non-zero on any blocking failure
make docker   # builds the image from requirements.lock, re-runs the EDA inside it, runs a sampling smoke test
```

Environment: Python 3.11 in `.venv`, pinned in `requirements.lock`. google-meridian
2.1.0, JAX 0.10.2, jaxlib 0.10.2, TensorFlow Probability
(tfp-nightly) 0.26.0.dev20260130. Backend: JAX (Meridian 2.x default; `MERIDIAN_BACKEND` is not set).
TensorFlow is installed only as a transitive dependency. Long runs are wrapped in `caffeinate` on
macOS. A full `make all` on a laptop CPU takes a few hours, mostly MCMC; saved models are reused.

## Data

[`geo_all_channels.csv`](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv),
downloaded by `scripts/download_data.py` from a pinned upstream commit and checked against a SHA-256
checksum. The raw file is read-only and never modified. It has 40 geos x 156 weeks
(2021-01-25 to 2024-01-15). KPI: conversions, with revenue per conversion, so outcomes
are in revenue. Paid media: Channel0-4 impressions and spend. Controls: competitor sales,
sentiment. Promo is a non-media treatment and Organic_channel0 is organic media.

## EDA

Full report: [reports/eda.md](reports/eda.md); Meridian's EDA:
[model_build/eda_default/eda.html](model_build/eda_default/eda.html).

- **Data quality:** the panel is balanced, with 0 missing values. Spend is
  impressions x a constant CPM in every channel. Spend shares:
  Channel0 18.5%, Channel1 14.3%, Channel2 5.5%, Channel3 40.0%, Channel4 21.7%.
- **Trend and seasonality:** a national trend of +1.46%/year
  (HAC p = 0.010), which a straight line describes (yearly breaks
  p = 0.10). No yearly seasonality
  (p = 0.28 vs simulated noise).
- **Noise floor:** geo-week revenue CV 30.0%; national CV
  6.9%; a 13-week moving average misses national revenue by
  5.1%. Lag-1 autocorrelation is about 0.
- **Weak media signal:** within-geo spend-revenue correlations are all |r| <= 0.08, with no clear lag
  decay. The controls confound spend and revenue (they correlate with spend up to
  0.55), and their
  effects vary strongly across geos (I2 94%
  and 88%).
- **Collinearity is low:** within-geo VIF is at most 2.11. Revenue and
  spend scale about 1:1 with population.

## Candidates and selection

Details: [reports/candidates.md](reports/candidates.md). Convergence gate: R-hat <= 1.05,
bulk/tail ESS >= 400, <= 0.5% divergent draws.

- **Round 1.** C1-C3: knots (AKS vs linear) x media pooling (default vs strong); default sampler. Result: C2 did not converge, C3 did not converge, C1 did not converge.
- **Round 2.** R1-R2: after round 1 failed in the baseline parameters, linear knots and strong pooling (the docs' convergence checklist), with max lag 4 vs 8; default sampler. Result: R1 converged, R2 converged.

Among the converged fits, PSIS-LOO with a 2-SE simplicity rule (R2 vs R1: ELPD +3.6 (SE 2.5, needs > 5.0)) keeps
**R1** (`linear_strong_pool_lag4`).

## Results

- **Diagnostics** ([reports/diagnostics.md](reports/diagnostics.md)): the selected model converges
  (max R-hat 1.0091). In-sample R2 is 0.77
  (geo-week) and 0.48 (national); 94% coverage is
  95% and 90%.
  Out-of-sample checks fail as described in the verdict. Meridian's own health check scores
  93.1/100 (PASS), using looser
  thresholds than our gates.
- **Measurement** ([reports/measurement.md](reports/measurement.md)): paid media drove
  255.9M of the 1,318.7M revenue
  (19.4%; 94% CI
  177.4M-364.1M). Highest mean ROI:
  Channel3 1.40 and Channel4 1.28;
  lowest: Channel0 0.76. Marginal ROI is below 1 for every channel.

  | channel | spend share | ROI (94% CI) | mROI | P(ROI>1) |
  |---|---|---|---|---|
  | Channel0 | 18.5% | 0.76 (0.18-1.81) | 0.34 | 0.24 |
  | Channel1 | 14.3% | 0.82 (0.25-1.67) | 0.43 | 0.27 |
  | Channel2 | 5.5% | 1.27 (0.39-2.54) | 0.69 | 0.64 |
  | Channel3 | 40.0% | 1.40 (0.60-2.34) | 0.76 | 0.81 |
  | Channel4 | 21.7% | 1.28 (0.27-3.24) | 0.49 | 0.57 |

- **Budget recommendations** ([reports/measurement.md](reports/measurement.md#budget-scenarios);
  Meridian reports [main](reports/meridian/optimization_main.html),
  [stress](reports/meridian/optimization_stress.html)): the total is held at the last 52 weeks'
  spend (71.4M).
  - Within +/-20% per channel: Channel0 -20%, Channel1 -20%, Channel2 +20%, Channel3 +20%, Channel4 -11%. Expected gain 1.8M (94% CI
    -1.7M to 5.1M), P(gain > 0) = 0.84.
  - Within +/-50%: Channel0 -49%, Channel1 -31%, Channel2 +18%, Channel3 +40%, Channel4 -13%. Expected gain 2.5M, P(gain > 0) =
    0.77.

  Moving spend toward Channel3 and Channel2 is more likely than not to help, but how much is uncertain.
  Beyond historically observed spend, the results assume the response-curve shape.
- **Calibration (HYPOTHETICAL lift test):** a Channel3 test reporting ROI
  1.4 narrows Channel3's ROI interval and leaves the other channels
  almost unchanged. The model implied 1.43 for the test geos and
  weeks, and the choice of test geos does not drive the result.

Channel-level file: `outputs/results/channel_results.csv`, with column definitions and package
versions in `channel_results_metadata.json`.

## Gates

`make gates`; details in [reports/diagnostics.md](reports/diagnostics.md#gate-summary-make-gates).

| gate | status | blocking |
|---|---|---|
| data_integrity | PASS | yes |
| prior_predictive_coverage | PASS | yes |
| convergence | PASS | yes |
| insample_r2 | PASS | yes |
| insample_coverage | PASS | yes |
| within_geo_r2 | PASS | no |
| residual_autocorrelation | PASS | no |
| cv_fold_104 | FAIL | yes |
| cv_fold_117 | FAIL | yes |
| cv_fold_130 | FAIL | yes |
| cv_holdout | FAIL | yes |
| roi_fold_stability | FAIL | yes |
| channel_roi_checks | WARN | no |

## What the data can't support

- Channel-level payback decisions: no channel has P(ROI > 1) >= 0.9 or <= 0.1.
- Geo-level ROI: not reported. Media effects are strongly pooled across geos by design.
- Forecast-grade claims: out-of-sample evidence comes from a single converged fold.
- Allocations far outside historical spend: these rely on the assumed response-curve shape.
- A precise baseline/media split: the two trade off almost one-for-one.

## Next steps

- Run a real lift test for Channel3 (the largest channel with undetermined payback) and replace
  the HYPOTHETICAL row in `data/lift_tests/lift_tests.csv`. The calibration pipeline is ready.
- Get more out-of-sample evidence. The fold fits need a sampler setup that converges reliably on
  shorter windows.
- Add a baseline signal that separates demand from media (e.g. category search volume) to
  reduce the baseline-media trade-off.
- Test lower-spend channels (Channel2, Channel4), whose ROI intervals are widest relative to spend.

## Repository layout

```
data/raw/                 downloaded CSV (gitignored, checksum-verified)
data/lift_tests/          HYPOTHETICAL lift test + file-format README
model_build/              Meridian skill scripts: build_model.py (EDA, fit, save),
                          results_report.py (health check, results summary),
                          run_budget_optimization.py; *.binpb models (gitignored)
scripts/                  EDA, selection, CV, gates, calibration, results, reports
outputs/                  JSON/CSV outputs behind every reported number
reports/                  eda.md, candidates.md, diagnostics.md, measurement.md, figures/,
                          meridian/ (Meridian HTML reports)
Makefile, Dockerfile, requirements.lock
```
