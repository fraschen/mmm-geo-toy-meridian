# Geo-level Bayesian MMM with Google Meridian

One production Meridian model (JAX backend) of Meridian's simulated geo dataset: 40 geos x
156 weeks, 5 paid channels, revenue as the outcome. This README covers the choices we
made, the diagnostics we trust, and what the model can and cannot tell a client.

## Verdict

**Not yet good enough for a client decision.** The model converges on the full data, fits
well in sample, and its in-sample intervals are well calibrated. Three things stop it from
being client-ready:

- **Out-of-sample evidence is thin.** Only 1 of 4 validation refits
  converged. In that one, national 94% interval coverage was 77%, below the 80% gate.
  ROI stability across refits could not be assessed.
- **No channel's payback is determined.** P(ROI > 1) ranges from 0.24 to
  0.81, and every 94% ROI interval spans 1.
- **Baseline and media are only weakly separated.** Total baseline and total media contribution
  correlate at -0.97 across posterior draws, so totals
  are firmer than channel splits.

Any number shared from this model should carry these caveats: national results only, wide
ROI intervals, thin out-of-sample validation, and budget gains whose credible intervals
include losses.

## The model

Defined in [`config/model.toml`](config/model.toml) and fitted by
[`model_build/build_model.py`](model_build/build_model.py) (Meridian's model-building skill):

- **Baseline:** a straight line over time (`knots = 2`). The EDA finds a linear trend and no
  yearly seasonality.
- **Media:** five paid channels with geometric adstock (max lag 4) and Hill saturation.
  Effects are strongly pooled across geos (`eta_m ~ HalfNormal(0.1)`), because each geo's own
  media signal is weak. Every channel gets the same ROI prior, `LogNormal(0.2, 0.9)`, so
  differences between channels come from the data.
- **Other inputs:** competitor sales and sentiment as controls (they confound spend and
  revenue), Promo, and organic media.
- **Scaling:** KPI and media are population-scaled.
- **Sampler:** NUTS with 4 chains x 1000 draws, seed 20260128.

**Why this model.** It was chosen once, from candidates that varied what the EDA cannot settle:
baseline flexibility, media pooling and max lag. The full record is in
[docs/model_selection](docs/model_selection/README.md); those candidates are not part of the pipeline.

- **Round 1:** C2, C3, C1 (an automatically selected 66-knot baseline or a
  linear one, with default or strong pooling) did not converge. The failures sat in the baseline.
- **Round 2:** two models with a linear baseline and strong pooling, with max lag 4 or 8, converged.
- **Comparison:** the lag-8 model's PSIS-LOO ELPD was higher by 3.6, with paired SE 2.5, short of the 5.0 needed to prefer the more complex model, so the lag-4 model was kept.
- The candidate set and the convergence criteria were revised after earlier rounds failed to converge; the original convergence gate was R-hat <= 1.01, bulk/tail ESS >= 400 and 0 divergences.

## Diagnostics we trust

Rules and thresholds: [`config/gates.toml`](config/gates.toml), documented in
[reports/gates.md](reports/gates.md). Full report: [reports/diagnostics.md](reports/diagnostics.md).

- **Convergence (blocking):** every element of every parameter must meet R-hat <= 1.05,
  bulk and tail ESS >= 400, and at most 0.5% divergent draws.
  The model has max R-hat 1.0091, min ESS 441 (bulk) and
  690 (tail), and 1 divergence out of 4000 draws.
  A fit that fails reports nothing else.
- **Out-of-sample forecasts (blocking, the decisive test):** the model is refitted on weeks 1-104,
  1-117, 1-130 and 1-143, each forecasting the next 13 weeks, with all scaling computed inside the
  refit. Only 1 refit converged, so this gate fails.
- **In-sample fit, judged against the noise floor:** R2 is 0.77 at geo-week level and
  0.48 nationally, and 94% coverage is 95% and 90%.
  National MAPE is 4.0%, against 5.1% for a
  13-week moving average. The national R2 looks low because that series barely varies (CV
  6.9%), not because the model misses it.
- **Identification:** ROI posteriors contract by 0.61-
  0.83 relative to the prior, so the data move every channel.
  The baseline-media correlation of -0.97 limits how far the channel
  splits can be trusted.
- **Not relied on:** Meridian's health score (93.1/100,
  PASS). It checks convergence only at R-hat < 1.2 and scores
  fit in sample only, so it cannot see the out-of-sample failure above.

## What the model can tell a client

From [reports/measurement.md](reports/measurement.md); file `outputs/results/channel_results.csv`,
with column definitions and package versions in `channel_results_metadata.json`.

- **Paid media in total** drove 255.9M of 1,318.7M revenue
  (19.4%; 94% CI 177.4M-364.1M).
- **Ranking of channels, with wide uncertainty:** highest mean ROI Channel3
  (1.40) and Channel4 (1.28); lowest
  Channel0 (0.76). Marginal ROI is below 1 for every channel, so
  spending more on any single channel is unlikely to pay back at the margin.

  | channel | spend share | ROI (94% CI) | mROI | P(ROI>1) |
  |---|---|---|---|---|
  | Channel0 | 18.5% | 0.76 (0.18-1.81) | 0.34 | 0.24 |
  | Channel1 | 14.3% | 0.82 (0.25-1.67) | 0.43 | 0.27 |
  | Channel2 | 5.5% | 1.27 (0.39-2.54) | 0.69 | 0.64 |
  | Channel3 | 40.0% | 1.40 (0.60-2.34) | 0.76 | 0.81 |
  | Channel4 | 21.7% | 1.28 (0.27-3.24) | 0.49 | 0.57 |

- **Direction of a reallocation** ([budget scenarios](reports/measurement.md#budget-scenarios); Meridian
  reports [main](reports/meridian/optimization_main.html), [stress](reports/meridian/optimization_stress.html)).
  The total is held at the last 52 weeks' spend (71.4M).
  - Within +/-20% per channel: Channel0 -20%, Channel1 -20%, Channel2 +20%, Channel3 +20%, Channel4 -11%. Expected gain 1.8M (94% CI
    -1.7M to 5.1M), P(gain > 0) = 0.84.
  - Within +/-50%: Channel0 -49%, Channel1 -31%, Channel2 +18%, Channel3 +40%, Channel4 -13%. Expected gain 2.5M, P(gain > 0) = 0.77.

  Moving spend toward Channel3 and Channel2 is more likely than not to help; the size of the gain is uncertain.

## What the model cannot tell a client

- **Whether a specific channel pays back:** no channel has P(ROI > 1) >= 0.9 or <= 0.1.
- **Forecasts:** the out-of-sample evidence comes from a single converged refit, and its national
  intervals were too narrow.
- **Geo-level ROI:** not reported; media effects are pooled across geos by design.
- **Spend far outside history:** results beyond observed spend assume the shape of the response curve.
- **A precise baseline/media split:** the two trade off almost one-for-one.

**Calibration (optional, HYPOTHETICAL).** `make calibration` refits the model with an invented
Channel3 lift test (ROI 1.4), to show how a real test would be used. It narrows
Channel3's ROI interval and leaves the other channels almost unchanged. The model implied
1.43 for the test geos and weeks, and the choice of test geos does not drive
the result. It is not part of the production model.

## Run it

```bash
make all      # production pipeline: data, EDA, fit, convergence gate, validation refits,
              # Meridian health check and summary, budget scenarios, gates, results, reports
make gates    # acceptance gates; exits non-zero on any blocking failure
make docker   # build the image from requirements.lock, re-run the EDA inside it, sampling smoke test
docker run --rm mmm-geo-toy-meridian   # the container's default command runs `make all`
```

- **Environment:** Python 3.11 in `.venv`, pinned in `requirements.lock`. google-meridian
  2.1.0, JAX 0.10.2, jaxlib 0.10.2, TensorFlow Probability
  (tfp-nightly) 0.26.0.dev20260130.
- **Backend:** JAX, Meridian 2.x's default; `MERIDIAN_BACKEND` is not set. TensorFlow is installed
  only as a transitive dependency.
- **Runtime:** fits are skipped when a saved model matches `config/model.toml`. A cold run is one
  full fit plus four validation refits, about 30 minutes of sampling on a laptop CPU.

## Data and EDA

[`geo_all_channels.csv`](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv)
is downloaded from a pinned upstream commit and checked against a SHA-256 checksum, and is never
modified. It has 40 geos x 156 weeks (2021-01-25 to 2024-01-15).
KPI: conversions x revenue per conversion. Full EDA: [reports/eda.md](reports/eda.md).

- **Panel and spend:** balanced panel with 0 missing values. Spend is
  impressions x a constant CPM. Spend shares: Channel0 18.5%, Channel1 14.3%, Channel2 5.5%, Channel3 40.0%, Channel4 21.7%.
- **Trend and seasonality:** trend +1.46%/year
  (HAC p = 0.010); no yearly seasonality (p = 0.28
  against simulated noise).
- **Noise floor:** geo-week revenue CV 30.0%, national 6.9%.
- **Media signal:** within-geo spend-revenue correlations are all |r| <= 0.08.
- **Controls:** they correlate with spend (up to 0.55),
  so they are confounders and stay in the model.
- **Collinearity:** within-geo VIF is at most 2.11.

## Gates

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

## Next steps

- **A real Channel3 lift test:** Channel3 is the largest channel with undetermined payback. Put the
  result in `data/lift_tests/lift_tests.csv` (format in `data/lift_tests/README.md`).
- **Reliable validation refits:** they need a sampler setup that converges on shorter windows,
  for example a higher target acceptance.
- **A baseline signal:** something that separates demand from media (e.g. category search
  volume) would reduce the baseline-media trade-off.

## Repository layout

```
config/model.toml         the production model (spec, priors, sampler, paths)
config/gates.toml         acceptance gates (rules, thresholds, rationale)
data/raw/                 downloaded CSV (gitignored, checksum-verified)
data/lift_tests/          HYPOTHETICAL lift test + file-format README
model_build/              Meridian skill scripts: build_model.py, results_report.py,
                          run_budget_optimization.py; fitted models (*.binpb, gitignored)
scripts/                  EDA, convergence gate, validation, gates, results, reports
outputs/                  JSON/CSV behind every reported number
reports/                  eda.md, diagnostics.md, measurement.md, gates.md, figures/,
                          meridian/ (Meridian HTML reports)
docs/model_selection/     how the model was chosen: candidates, diagnostics, comparison
Makefile, Dockerfile, requirements.lock
```
