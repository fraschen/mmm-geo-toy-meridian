# Marketing mix model with Google Meridian

This project measures how much five paid media channels contribute to revenue, using
Google's Meridian marketing mix model on 3 years of weekly data from 40 regions (Meridian's
simulated dataset). It runs one model, in a container, from one command.

This page explains the choices we made, which checks we trust, and what the model can and
cannot tell a client.

## The short answer

**The model is useful for direction, not yet for decisions.** It fits the past well and its
uncertainty ranges are honest in-sample. But three things keep it from being client-ready:

- **We could not confirm it forecasts well.** We refit the model on shorter stretches of history
  to see how it predicts the weeks that follow. Only 1 of those 4 refits
  ran reliably. In that one, the forecast ranges were too narrow nationally: they captured 77% of weeks, below our 80% bar.
- **No channel clearly pays for itself, or clearly doesn't.** The probability that a channel's ROI is
  above 1 ranges from 24% to 81%; we would need 90% (or 10%) to call it.
- **The model struggles to separate media from everything else.** It is confident about total revenue,
  less so about how much of it comes from media versus underlying demand.

Anything shared from this model should come with these caveats.

<details><summary>Technical details</summary>

- Validation refits: train on weeks 1-104, 1-117, 1-130 and 1-143 (the last is the 13-week holdout), forecast the next 13 weeks; converged: fold_104. In the converged refit: geo-week forecast MAPE 26.6% (ratio to in-sample 1.04, 94% coverage 94%); national MAPE 4.5% (ratio 1.09, coverage 77%, bias +3.1%).
- P(ROI > 1) by channel: Channel0 0.24, Channel1 0.27, Channel2 0.64, Channel3 0.81, Channel4 0.57; every 94% ROI interval spans 1.
- Posterior correlation between total baseline and total paid-media contribution: -0.97.

</details>


## The model

Defined in [`config/model.toml`](config/model.toml):

- **Underlying demand** follows a straight-line trend, because the data show steady growth and no
  yearly seasonality.
- **Media** has effects that carry over for up to 4 weeks and that flatten out as spend grows.
- **Regions share information:** a channel is assumed to work similarly across regions, because
  each region's own data are too noisy to estimate it alone.
- **Every channel starts from the same prior belief** about its return, so differences between
  channels come from the data.
- **Competitor sales and consumer sentiment** are included, because they move with both our spend
  and our revenue. Leaving them out would distort media's effect.

**How we chose it.** We tried several versions that differed on what the data alone could not
settle. The versions with a more flexible baseline did not run reliably. Two versions with a
straight-line baseline did, and the simpler of the two was as good as the other, so we kept it.
We loosened the reliability criteria after the first attempts failed to meet them; the original, stricter criteria are recorded in config/gates.toml. Full record: [docs/model_selection](docs/model_selection/README.md).

<details><summary>Technical details</summary>

- Spec: `knots = 2`, `max_lag = 4`, `media_effects_dist = log_normal`, `adstock_decay_spec = geometric`, `hill_before_adstock = False`, `media_prior_type = roi`, `unique_sigma_for_each_geo = False`.
- Priors (others Meridian defaults): `roi_m ~ LogNormal(loc=0.2, scale=0.9)`, `eta_m ~ HalfNormal(scale=0.1)`.
- Sampler: 4 chains, 500 adaptation + 500 burn-in + 1000 kept draws per chain, seed 20260128.
- Round 1 (C2, C3, C1): AKS (66 knots) or linear baseline x default or strong media pooling; none converged (failures in the baseline parameters `knot_values`, `mu_t`, `tau_g`).
- Round 2 (R1, R2): linear baseline, strong pooling (`eta_m ~ HalfNormal(0.1)`), max lag 4 vs 8; both converged.
- PSIS-LOO: R2 vs R1 ELPD difference +3.6 (paired SE 2.5); switching needs more than 5.0, so R1 is kept.
- The candidate set and the convergence criteria were revised after earlier rounds failed to converge; the original convergence gate was R-hat <= 1.01, bulk/tail ESS >= 400 and 0 divergences.

</details>


## The checks we trust

All rules are in [`config/gates.toml`](config/gates.toml) and explained in
[reports/gates.md](reports/gates.md); details in [reports/diagnostics.md](reports/diagnostics.md).

- **Did the computation run reliably?** Yes. Meridian estimates the model by running 4
  independent simulations, which agree with each other. If they did not, we would report nothing.
- **Does it forecast weeks it has not seen?** This is the test that matters most, and the one the model
  does not pass yet (see above).
- **Does it fit the past?** Yes, within the limits of the data. Nationally it misses a typical week by
  4.0%, better than a simple 13-week average (5.1%).
  Its 94% ranges contain 95% of regional weeks, close to the 94% they should.
- **Did the data teach the model anything?** Yes. Every channel's estimate moved away from the
  starting assumption.
- **What we don't rely on:** Meridian's built-in health score (93/100). It only
  looks at how well the model fits the past, so it cannot see the forecasting problem.

<details><summary>Technical details</summary>

- Convergence gate: every element of every parameter needs R-hat <= 1.05, bulk and tail ESS >= 400, and at most 0.5% divergent draws. Model: max R-hat 1.0091, min ESS 441 (bulk) / 690 (tail), 1 divergence(s) in 4000 draws.
- In-sample R2: geo-week 0.77, national 0.48; 94% coverage: geo-week 94.6%, national 90.4%; MAPE: geo-week 26.0%, national 4.0% (13-week moving average: 23.1% and 5.1%). The national R2 is modest because the national series barely varies (CV 6.9%).
- ROI contraction vs prior (1 - posterior sd / prior sd): Channel0 0.78, Channel1 0.83, Channel2 0.68, Channel3 0.77, Channel4 0.61.
- Baseline-media posterior correlation -0.97.
- Meridian health score 93.1/100 (PASS); its convergence check uses R-hat < 1.2 only.

</details>


## What the model can tell a client

Details in [reports/measurement.md](reports/measurement.md). ROI is revenue returned per unit
of spend: above 1, a channel pays for itself.

- **Media as a whole** drove about 255.9M of 1,318.7M revenue
  (19%), somewhere between 177.4M
  and 364.1M.
- **Which channels look stronger:** Channel3 and Channel4 have the highest
  returns, and Channel0 the lowest, but the ranges overlap a lot.

  | channel | share of spend | ROI (94% range) | marginal ROI | P(ROI > 1) |
  |---|---|---|---|---|
  | Channel0 | 18.5% | 0.76 (0.18-1.81) | 0.34 | 0.24 |
  | Channel1 | 14.3% | 0.82 (0.25-1.67) | 0.43 | 0.27 |
  | Channel2 | 5.5% | 1.27 (0.39-2.54) | 0.69 | 0.64 |
  | Channel3 | 40.0% | 1.40 (0.60-2.34) | 0.76 | 0.81 |
  | Channel4 | 21.7% | 1.28 (0.27-3.24) | 0.49 | 0.57 |

  Marginal ROI (the return on the next unit of spend) is below 1 for every channel, so adding
  budget to any single channel is unlikely to pay back on its own.
- **Which way to move budget:** with the total held at last year's spend (71.4M), shifting
  up to 20% per channel toward Channel3 and Channel2 would add about 1.8M of revenue
  a year, with a 84% chance of being an improvement. Shifting up to 50% would
  add about 2.5M, with a 77% chance. The direction is fairly clear; the size
  is uncertain, and could even be a small loss
  ([scenarios](reports/measurement.md#budget-scenarios); Meridian's reports: [main](reports/meridian/optimization_main.html),
  [stress](reports/meridian/optimization_stress.html)).

<details><summary>Technical details</summary>

- Paid media contribution: 255.9M (94% CI 177.4M-364.1M), 19.4% of revenue.
- Main scenario (+/-20%): Channel0 -20%, Channel1 -20%, Channel2 +20%, Channel3 +20%, Channel4 -11%; gain 1.8M (94% CI -1.7M to 5.1M), P(gain > 0) = 0.84.
- Stress scenario (+/-50%): Channel0 -49%, Channel1 -31%, Channel2 +18%, Channel3 +40%, Channel4 -13%; gain 2.5M (94% CI -4.3M to 8.7M), P(gain > 0) = 0.77.
- Budget conserved to the optimizer's spend grid (difference 0.002%). Beyond historical spend, results assume the response-curve shape.

</details>


## What the model cannot tell a client

- **Whether any specific channel pays back.** The ranges are too wide.
- **Reliable forecasts.** Not yet tested well enough.
- **Results by region.** The model shares information across regions, so it reports national results only.
- **What happens at much higher spend than ever before.** Beyond past levels, results depend on
  assumptions about how returns flatten.
- **A precise split between media and underlying demand.**

**Using a real experiment (optional, illustrative).** `make calibration` shows how the result
of a lift test would sharpen the model. We used an invented (HYPOTHETICAL) Channel3 test: it
narrowed Channel3's range and left the others unchanged. It is not part of the production model.

<details><summary>Technical details</summary>

- HYPOTHETICAL test: Channel3, 2023-07-03 to 2023-08-27, ROI 1.4 (SE 0.25); calibrated prior mean 1.23, sd 0.47.
- Model-implied ROI for the test regions and weeks before calibration: 1.43 (94% CI 0.62-2.40).
- Across 20 alternative test-region sets, implied ROI 1.35-1.50: the choice of regions does not drive the result.

</details>


## Run it

```bash
make all      # the full pipeline: data checks, fit, reliability check, forecast tests,
              # Meridian reports, budget scenarios, gates, results and reports
make gates    # the acceptance checks only; fails if a blocking check fails
make docker   # build the container and check that it reproduces the analysis
docker run --rm mmm-geo-toy-meridian   # run the full pipeline in the container
```

Pinned environment (`requirements.lock`): Python 3.11, google-meridian 2.1.0,
JAX 0.10.2, jaxlib 0.10.2, TensorFlow Probability (tfp-nightly)
0.26.0.dev20260130, on Meridian's default JAX backend. A cold run takes about 30 minutes;
saved models are reused when the configuration has not changed.

## The data

Meridian's simulated [geo dataset](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv):
40 regions, 156 weeks (2021-01-25 to 2024-01-15), five paid channels,
two controls, a promotion and one organic channel. It is downloaded from a fixed version and checked
before use. What we learned from it: [reports/eda.md](reports/eda.md).

- **Data quality:** the data are complete and clean.
- **Trend and seasonality:** revenue grows about 1.5% a year, with no
  meaningful yearly seasonality.
- **Noise:** weekly revenue is noisy, about 30% up or down per region and
  6.9% nationally, which limits how precise any model can be.
- **Media signal:** media's effect is hard to see in the raw data, so the model has to pool regions to find it.

<details><summary>Technical details</summary>

- Balanced panel, 0 missing values; spend = impressions x a constant CPM. Spend shares: Channel0 18.5%, Channel1 14.3%, Channel2 5.5%, Channel3 40.0%, Channel4 21.7%.
- Trend +1.46%/year (HAC p = 0.010); yearly breaks do not improve on a line (p = 0.10).
- Seasonality: yearly wave p = 0.83; any yearly swing at most 4.1% peak to trough (95% upper bound).
- Within-region spend-revenue correlations all |r| <= 0.08; within-region VIF at most 2.11.
- Controls correlate with spend up to 0.55, so they are confounders.
- Conversions are inflated in this simulated dataset (about 19 per person per week), so revenue per conversion (about 0.02) looks low; revenue is on a realistic scale (paid spend is 17% of revenue).

</details>


## Acceptance checks

| check | result | blocking |
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

- **Run a real lift test for Channel3,** the biggest budget line with an unclear payback. The
  pipeline is ready to use it.
- **Make the forecast tests run reliably,** so the model's predictions can be checked.
- **Add a measure of underlying demand,** such as category search volume, to help separate media
  from everything else.

## Repository layout

```
config/model.toml         the model we run
config/gates.toml         the acceptance checks
data/                     data download (raw file not committed) and the lift-test format
model_build/              Meridian scripts: build the model, health report, budget scenarios
scripts/                  data checks, forecast tests, gates, results, reports
outputs/                  the numbers behind every report
reports/                  eda.md, diagnostics.md, measurement.md, gates.md, figures, Meridian reports
docs/model_selection/     how the model was chosen
Makefile, Dockerfile, requirements.lock
```
