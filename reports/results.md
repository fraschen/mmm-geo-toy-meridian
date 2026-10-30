# Results

This report has three parts: **diagnostics** (can we trust the model?), **measurement** (what does it say?) and the **acceptance gates** (the rules the model must pass). What we learned about the data before modelling is in [eda.md](eda.md); the model is defined in [`config/model.toml`](../config/model.toml).

The numbers come from `outputs/` (`make gates`, `make validate`, `make results`, `make budget`, `make calibration`) and the charts from `make reports`.

# Part 1: Diagnostics

The production model is a geo-level Meridian MMM with:
- a straight-line baseline (`knots = 2`, one knot at each end);
- media effects strongly pooled across geos (`eta_m ~ HalfNormal(0.1)`);
- geometric adstock with a maximum lag of 4 weeks;
- ROI priors `LogNormal(0.2, 0.9)` on all paid channels.

Meridian's own reports: [health check](meridian/health_check.html), [Model Results Summary](meridian/results_summary.html).

## Why this model

We run one model. Its choices follow from what the EDA could and could not settle, and from what sampled reliably in earlier attempts:

- **Straight-line baseline:** the trend is roughly linear and there is no yearly seasonality.
- **Media effects strongly pooled across geos:** each geo's own media signal is too weak to estimate a separate effect.
- **Max lag 4:** the data do not say how long media effects last, and shorter lags sample more reliably.
- In earlier attempts, a more flexible baseline (Meridian's automatic knot selection) and looser pooling did not sample reliably.

## Are the inputs and starting assumptions sound?

Both pass.

- **Data:** the file matches its checksum, every geo has all 156 weeks, and nothing is missing, duplicated or impossible. Details in [eda.md](eda.md).
- **Starting assumptions:** before seeing the data, the model's 94% range contains all national weeks. That range is very wide (about 7 times average weekly revenue), so this only rules out starting assumptions that could not produce the data at all.

## Did the computation run reliably? (convergence)

Yes. Meridian runs 4 independent simulations; if they agree, the results can be trusted.

- Gate: R-hat ≤ 1.05, bulk and tail ESS ≥ 400, at most 20 divergent draws out of 4,000.
- Model: max R-hat 1.009, min bulk ESS 441, min tail ESS 690, 1 divergence. **PASS.**

![](figures/diagnostics/convergence.png)

## How well does the model fit the past?

Well, judged against how noisy weekly revenue is.

- **Accuracy:** R² is 0.77 per geo-week and 0.48 for national weekly revenue. MAPE is 26.0% per geo-week and 4.0% nationally. For comparison, a simple 13-week moving average misses by 23.1% per geo and 5.1% nationally.
- **Uncertainty ranges:** the 94% ranges contain 94.6% of geo-weeks and 90.4% of national weeks.
- **Within geos:** the model picks up week-to-week movements within each geo (within-geo R² 0.27), not only the differences between geos.
- **Misses look random:** residual lag-1 autocorrelation is +0.011 nationally and +0.001 for the median geo.

![](figures/diagnostics/insample_fit.png)

![](figures/diagnostics/residuals.png)

## Can it forecast weeks it has not seen? (holdout and CV)

Mostly yes, but the final holdout could not be checked.

For each fold the model is fitted with 13 weeks held out, using Meridian's holdout: the revenue of those weeks is left out of the fit. The held-out weeks follow weeks 104, 117, 130 and 143; the last fold is the final 13-week holdout. Accuracy is Meridian's `predictive_accuracy` on the training and held-out weeks.

The forecasts use the actual media, controls, Promo and organic of the held-out weeks. That is realistic for our own spend, but in a real forecast competitor sales and sentiment would not be known in advance, so this test is somewhat easier than real forecasting. Meridian also computes its revenue scaling from all weeks, held-out ones included. That is a small leak, and it made no measurable difference: on fold_104, the only fold that converged under both, the old approach (fitting truncated data with its own scaling) gave the same errors to three decimals.

| fold | train weeks | converged | geo MAPE (forecast / in-sample) | national MAPE (forecast / in-sample) | national bias | gate |
|---|---|---|---|---|---|---|
| fold_104 | 1-104 | yes | 26.6% / 25.6% | 4.5% / 4.1% | +3.1% | PASS |
| fold_117 | 1-117 | yes | 27.9% / 25.5% | 4.2% / 4.1% | −0.1% | PASS |
| fold_130 | 1-130 | yes | 26.9% / 25.7% | 2.5% / 4.1% | +0.2% | PASS |
| holdout | 1-143 | no | – | – | – | FAIL |

- **Three of four folds converged.** In each, forecast errors are close to the in-sample errors: the forecast-to-in-sample ratio is at most 1.09, against a limit of 1.25. National MAPE is 2.5–4.5%, against a limit of 10%.
- **The final holdout did not converge** (min bulk ESS 68, against a minimum of 400), so it gives no forecast evidence and fails the gate. Sampling on shorter or differently shaped windows is still fragile.
- **Which folds converge varies between runs.** Before the switch to Meridian's holdout, only fold_104 converged; after it, fold_117 and fold_130 passed and the final holdout got worse. The failure is in the baseline's level parameters (knot values, geo levels, control effects), which trade off against each other; the media estimates agreed across the four simulations.
- **The forecast drifted high only in fold_104,** by 3.1%. The later folds show no bias.

![](figures/diagnostics/cv_forecast.png)

**ROI is stable across folds.** In the three converged folds, each channel's mean ROI varies by at most 1.35x (Channel4: 1.46 to 1.98), well inside the 2x warning level. **PASS.**

![](figures/diagnostics/roi_by_fold.png)

## Can the model tell media apart from everything else?

Partly.

- **The data informed every ROI.** Each channel's ROI range is 61–83% narrower than the starting assumption (Channel0 0.78, Channel1 0.83, Channel2 0.68, Channel3 0.77, Channel4 0.61; the warning level is 0.30).
- **But the split between media and underlying demand is uncertain.** Across posterior draws, the total baseline and total paid-media contribution have a correlation of −0.97. The model pins down their sum precisely but not how it splits between them. This is the main source of ROI uncertainty.

![](figures/diagnostics/roi_prior_posterior.png)

## Meridian's own health checks

All pass (health score 93.1/100), but they are more lenient than our gates and only look at the past:

| check | status | compared with our gates |
|---|---|---|
| Convergence | PASS | Meridian requires R-hat < 1.2; we require ≤ 1.05, plus ESS and divergences |
| Baseline | PASS | |
| BayesianPPP | PASS | Meridian checks only total revenue; we check weekly interval coverage |
| GoodnessOfFit | PASS | Meridian reports R² and MAPE without a pass/fail bar; our in-sample and CV gates have explicit limits |
| PriorPosteriorShift | PASS | |
| ImplausibleROI | PASS | |
| HighVariance | PASS | |
| PotentialBias | PASS | |

## Known limitations

- **The final holdout fit did not converge**, so the most recent 13 weeks have no forecast check.
- **Baseline and media trade off almost one-for-one,** so channel ROIs are wide and no channel's payback is determined: P(ROI > 1) is between 0.1 and 0.9 for every channel.
- **The straight-line baseline cannot absorb week-level shocks common to all geos.**

# Part 2: Measurement

National results per channel over all 156 weeks (2021-01-25 to 2024-01-15). Revenue is conversions × revenue per conversion. Read Part 1 for the caveats.

Meridian's own reports: [Model Results Summary](meridian/results_summary.html), budget optimization [main](meridian/optimization_main.html) and [stress](meridian/optimization_stress.html). The channel table is also in `outputs/05_report/channel_results.csv`, with its metadata in `channel_results_metadata.json`.

## Channel results

For each channel: its contribution, ROI and marginal ROI (the return on the next unit of spend), with 94% ranges, and the probability that its ROI is above 1.

| channel | spend | spend share | contribution (94% CI) | share of revenue | ROI (94% CI) | mROI (94% CI) | P(ROI>1) | gate |
|---|---|---|---|---|---|---|---|---|
| Channel0 | 40.5M | 18.5% | 30.8M (7.3M-73.3M) | 2.3% | 0.76 (0.18-1.81) | 0.34 (0.06-0.80) | 0.24 | WARN |
| Channel1 | 31.3M | 14.3% | 25.7M (7.8M-52.5M) | 1.9% | 0.82 (0.25-1.67) | 0.43 (0.11-0.90) | 0.27 | WARN |
| Channel2 | 12.1M | 5.5% | 15.4M (4.7M-30.6M) | 1.2% | 1.27 (0.39-2.54) | 0.69 (0.20-1.41) | 0.64 | WARN |
| Channel3 | 87.9M | 40.0% | 123.0M (52.3M-205.9M) | 9.3% | 1.40 (0.60-2.34) | 0.76 (0.32-1.22) | 0.81 | WARN |
| Channel4 | 47.7M | 21.7% | 61.1M (12.7M-154.8M) | 4.6% | 1.28 (0.27-3.24) | 0.49 (0.10-1.00) | 0.57 | WARN |

- **Paid media drove 255.9M of revenue** (94% CI 177.4M-364.1M), 19.4% of the 1,318.7M observed. The baseline accounts for 1,012.9M.
- **Channel3 (1.40) and Channel4 (1.28) have the highest mean ROI, Channel0 (0.76) the lowest.** All ranges are wide, and no channel's payback is determined.
- **Marginal ROI is below ROI for every channel** (diminishing returns).
- **Cost per incremental conversion is not reported,** because results are in revenue.

![](figures/measurement/roi_mroi.png)

## Model fit and baseline

The model's weekly revenue against the actual, and the baseline: what revenue would have been with no media, Promo or organic activity.

![](figures/measurement/fit_baseline.png)

## Where revenue comes from

Total revenue broken down into the baseline and each driver's contribution.

![](figures/measurement/waterfall.png)

## Spend share vs contribution share

Channel2, Channel3 and Channel4 have a larger share of paid-media contribution than of spend: they do more than their budget suggests.

![](figures/measurement/share.png)

## Weekly contribution by channel

How much revenue each channel drove, week by week.

![](figures/measurement/weekly_contribution.png)

## Diminishing returns (saturation curves)

How each channel's effect flattens as its activity grows, before (grey) and after (colour) seeing the data, over the range of activity actually observed.

![](figures/measurement/saturation.png)

## Revenue vs spend (response curves)

How much revenue each channel would drive at different spend levels. Beyond the dashed line (historical spend), the curve is an assumption, not an observation.

![](figures/measurement/response_curves.png)

## Budget scenarios

How to split the same total budget to get more revenue, allowing each channel to move by up to 20% (main) or 50% (stress). The budget is fixed at the last 52 weeks' spend (71.4M), optimized over 2023-01-23 to 2024-01-15. The gain is the extra revenue at the optimal split compared with the historical split, computed for each posterior draw.

| scenario | limit per channel | change vs last 52 weeks | gain (94% CI) | P(gain > 0) |
|---|---|---|---|---|
| main | ±20% | Channel0 −20%, Channel1 −20%, Channel2 +20%, Channel3 +20%, Channel4 −11% | 1.8M (−1.7M to 5.1M) | 0.84 |
| stress | ±50% | Channel0 −49%, Channel1 −31%, Channel2 +18%, Channel3 +40%, Channel4 −13% | 2.5M (−4.3M to 8.7M) | 0.77 |

- **Main:** move spend towards Channel2 and Channel3. The expected gain is 1.8M (2.1% of media-driven revenue), and the shift is more likely than not to help (P = 0.84). But the range includes losses, so the size of the gain is uncertain.
- **Stress:** larger shifts give a larger expected gain (2.5M) but a wider range and a lower chance of helping (P = 0.77).
- **Beyond past spend levels, results rely on the assumed shape of the response curve.** Raising the last 52 weeks' spend by 50% would push 3–10% of geo-weeks per channel above each geo's highest-ever spend.

![](figures/measurement/budget.png)

## Using a lift test

How a real experiment would sharpen the model. The test here is invented, to show the method.

The lift test (`data/lift_tests/lift_tests.csv`, schema in `data/lift_tests/README.md`) covers Channel3, 2023-07-03 to 2023-08-27, in 10 test geos, with ROI 1.4 (SE 0.25). Meridian's `CalibrationBuilder` turns it into a Channel3 ROI prior with mean 1.23 and sd 0.47. The prior is wider than the test's own SE because the test covered only part of the channel's national spend. The calibrated refit converged (max R-hat 1.009, min bulk ESS 607, 0 divergences).

| channel | ROI before (94% CI) | P(ROI>1) before | ROI after (94% CI) | P(ROI>1) after |
|---|---|---|---|---|
| Channel0 | 0.76 (0.18-1.81) | 0.24 | 0.77 (0.19-1.88) | 0.25 |
| Channel1 | 0.82 (0.25-1.67) | 0.27 | 0.84 (0.27-1.68) | 0.28 |
| Channel2 | 1.27 (0.39-2.54) | 0.64 | 1.27 (0.40-2.48) | 0.66 |
| Channel3 | 1.40 (0.60-2.34) | 0.81 | 1.30 (0.73-2.01) | 0.81 |
| Channel4 | 1.28 (0.27-3.24) | 0.57 | 1.31 (0.28-3.17) | 0.59 |

- **Channel3's range narrows from 0.60–2.34 to 0.73–2.01; the other channels barely change.**
- **For the test geos and weeks, the uncalibrated model implied a Channel3 ROI of 1.43** (94% CI 0.62-2.40), close to the test's 1.4. After calibration it implies 1.33.
- **The test's 1.4 equals the model's national Channel3 ROI by coincidence** (the test value was fixed first), so calibration narrows the range more than it moves the mean.

![](figures/measurement/calibration.png)

## Caveats

Read these before sharing any number from this page.

- **Read Part 1 first:** the final holdout could not be checked.
- **Totals are firmer than channel splits:** baseline and media trade off almost one-for-one.
- **National results only:** there is no geo-level ROI.
- **The lift test was made up to show the method:** the calibrated numbers are not real evidence.

# Part 3: Acceptance gates

These are the checks the model must pass before its numbers go to a client. If a **blocking** gate fails, the model is not client-ready; a non-blocking gate only adds a caveat. The exact rules and thresholds are in [`config/gates.toml`](../config/gates.toml), and the latest results in `outputs/03_validate/gates.json`. `make gates` stops with an error if any blocking gate fails.

The convergence criteria were revised after earlier rounds failed to converge. The original gate was R-hat ≤ 1.01, bulk and tail ESS ≥ 400, and 0 divergences.

**One blocking gate fails: `cv_holdout`.**

| gate | the question it answers | blocking | result |
|---|---|---|---|
| data_integrity | Is the data file the one we expect, complete and free of impossible values? | yes | PASS |
| prior_predictive_coverage | Before seeing the data, can the model's starting assumptions produce numbers like the real ones? | yes | PASS |
| convergence | Did the computation run reliably? | yes | PASS |
| negative_baseline | Is the baseline positive, as it must be? (probability of a negative baseline ≤ 0.2) | yes | PASS |
| insample_r2 | Does the model explain a reasonable share of what happened? | yes | PASS |
| insample_coverage | Do the 94% ranges contain about 94% of past weeks? | yes | PASS |
| within_geo_r2 | Does the model pick up week-to-week movements within each geo? | no | PASS |
| residual_autocorrelation | Are the model's misses random? | no | PASS |
| cv (fold_104, fold_117, fold_130, holdout) | Does the model forecast 13 weeks it has not seen? | yes | PASS, PASS, PASS, **FAIL** (holdout fit did not converge) |
| roi_fold_stability | Do channel ROIs stay roughly the same when the model is fitted on less data? | yes | PASS |
| channel_roi_checks | For each channel: did the data inform the ROI, and is it clear whether it pays back? | no | WARN (no channel's payback is determined) |
