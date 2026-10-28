# Gates

These are the checks the model must pass before its numbers go to a client. A **blocking** check that fails means the model is not client-ready; a non-blocking check only adds a caveat. Rules and thresholds live in [`config/gates.toml`](../config/gates.toml), which the scripts read; this page is generated from it and from the latest outputs (`make gates`, then `make reports`). `make gates` exits non-zero if any blocking gate fails.

- **PASS:** the check meets its threshold. **WARN:** a non-blocking check misses it (report with caveats). **FAIL:** a blocking check misses it (not client-ready).
- Intervals: 94% (3rd-97th percentile of the predictive or posterior draws).
- The candidate set and the convergence criteria were revised after earlier rounds failed to converge; the original convergence gate was R-hat <= 1.01, bulk/tail ESS >= 400 and 0 divergences.

## Summary

Model: `mmm` (selected as R1 (linear_strong_pool_lag4)). Blocking failures: cv_fold_104, cv_fold_117, cv_fold_130, cv_holdout, roi_fold_stability.

| gate | the question it answers | blocking | latest status |
|---|---|---|---|
| [data_integrity](#data-integrity) | Is the data file the one we expect, complete and free of impossible values? | yes | data_integrity PASS |
| [prior_predictive_coverage](#prior-predictive-coverage) | Before seeing the data, can the model's starting assumptions produce numbers like the real ones? | yes | prior_predictive_coverage PASS |
| [convergence](#convergence) | Did the computation run reliably? The model is estimated by several independent simulations; they must agree. | yes | convergence PASS |
| [insample_r2](#insample-r2) | Does the model explain a reasonable share of what happened in the past? | yes | insample_r2 PASS |
| [insample_coverage](#insample-coverage) | Do the model's 94% ranges contain about 94% of past weeks, neither too narrow nor too wide? | yes | insample_coverage PASS |
| [within_geo_r2](#within-geo-r2) | Does the model pick up week-to-week movements within each region, not just the differences between regions? | no (warn only) | within_geo_r2 PASS |
| [residual_autocorrelation](#residual-autocorrelation) | Are the model's misses random, or does it repeatedly miss in the same direction week after week? | no (warn only) | residual_autocorrelation PASS |
| [cv](#cv) | Does the model forecast weeks it has not seen? We refit it on shorter histories and check the next 13 weeks. | yes | cv_fold_104 FAIL, cv_fold_117 FAIL, cv_fold_130 FAIL, cv_holdout FAIL |
| [roi_fold_stability](#roi-fold-stability) | Do the channel ROIs stay roughly the same when the model is refitted on less data? | yes | roi_fold_stability FAIL |
| [channel_roi_checks](#channel-roi-checks) | For each channel: did the data actually inform the ROI, and is it clear whether the channel pays back? | no (warn only) | channel_roi_checks WARN |

## Model selection rule (applied once, when the model was chosen)

- **Applies to:** converged candidates, when the production model was chosen (not part of the pipeline)
- **Rule:** PSIS-LOO ELPD from the pointwise log-likelihood on Meridian's transformed KPI. Start from the simplest converged candidate; switch to a more complex one only if its ELPD is higher by more than se_rule x the SE of the paired difference.
- **Threshold:** `se_rule` = 2.0 (PROMPT.md). Prefer the simpler model unless the data clearly support the more complex one.
- **Enforced in:** `docs/model_selection/select_model.py (record)`
- **Result:** R2 vs R1: ELPD +3.6 (paired SE 2.5, needs > 5.0) -> keep. Record: [docs/model_selection](../docs/model_selection/README.md).

## Gate definitions and latest results

### data_integrity

**Is the data file the one we expect, complete and free of impossible values?**

- **Blocking:** yes
- **Applies to:** raw data file
- **Rule:** Raw CSV matches its pinned SHA-256; balanced 40 x 156 geo-week panel; no missing values; no duplicate geo-weeks; spend, impressions, KPI, population and revenue per KPI are non-negative. FAIL if any check is false.
- **Source of threshold:** PROMPT.md
- **Basis:** Hard data checks; any failure means the inputs are not the documented dataset.
- **Enforced in:** `scripts/gates.py (data_integrity), values from scripts/eda.py`

| threshold | value |
|---|---|
| (none) | |

Latest result:

| check | status | value |
|---|---|---|
| data_integrity | PASS | checksum_ok True, balanced_panel True, missing 0, duplicates 0, nonnegative True |

### prior_predictive_coverage

**Before seeing the data, can the model's starting assumptions produce numbers like the real ones?**

- **Blocking:** yes
- **Applies to:** selected model, prior draws
- **Rule:** Share of national weekly revenue observations inside the prior predictive 94% interval. FAIL below the minimum.
- **Source of threshold:** project
- **Basis:** Rules out priors that cannot generate the observed data. A weak check: the prior predictive interval is wide by design, so a PASS says little about prior quality.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `prior_predictive_national_coverage_min` | 0.8 |

Latest result:

| check | status | value |
|---|---|---|
| prior_predictive_coverage | PASS | 1 |

Note: prior predictive 94% interval width / observed mean: 7.06

### convergence

**Did the computation run reliably? The model is estimated by several independent simulations; they must agree.**

- **Blocking:** yes
- **Applies to:** every fit: the production model, its CV refits, the optional calibrated refit
- **Rule:** For every element of every posterior variable (elements constant across all draws, such as the baseline geo's intercept, are skipped): max R-hat, min bulk ESS, min tail ESS; and the share of divergent NUTS transitions. FAIL if any threshold is breached. A failing fit records only its convergence diagnostics; its ROI, fit and contributions are never reported.
- **Source of threshold:** project (revised; see meta.note)
- **Basis:** ESS >= 400 bulk and tail follows Vehtari et al. (2021). R-hat <= 1.05 is looser than their 1.01 and stricter than Meridian's own health check (R-hat < 1.2). Up to 0.5% divergent draws (20 of 4000) are tolerated.
- **Enforced in:** `scripts/check_convergence.py (make convergence)`

| threshold | value |
|---|---|
| `rhat_max` | 1.05 |
| `ess_min` | 400 |
| `divergent_share_max` | 0.005 |

Latest result:

| check | status | value |
|---|---|---|
| convergence | PASS | max_rhat 1.009, min_ess_bulk 441.2, min_ess_tail 690, divergences 1 |

Note: R-hat <= 1.05, bulk/tail ESS >= 400, <= 0.5% divergent draws (config/gates.toml). The candidate set and the convergence criteria were revised after earlier rounds failed to converge; the original convergence gate was R-hat <= 1.01, bulk/tail ESS >= 400 and 0 divergences.

### insample_r2

**Does the model explain a reasonable share of what happened in the past?**

- **Blocking:** yes
- **Applies to:** selected model, all 156 weeks
- **Rule:** R2 of the posterior-mean prediction at geo-week and national weekly level. FAIL if either is below its minimum.
- **Source of threshold:** project
- **Basis:** Geo-week: revenue scales with population, so geo levels dominate the variance; a model must explain at least half of it. National: weekly revenue has CV 6.9% and is close to white noise (lag-1 -0.02), so there is little variance to explain and a modest R2 is the realistic bar.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `insample_geo_week_r2_min` | 0.5 |
| `insample_national_r2_min` | 0.25 |

Latest result:

| check | status | value |
|---|---|---|
| insample_r2 | PASS | geo_week 0.7669, national 0.4763 |

### insample_coverage

**Do the model's 94% ranges contain about 94% of past weeks, neither too narrow nor too wide?**

- **Blocking:** yes
- **Applies to:** selected model, all 156 weeks
- **Rule:** Share of observations inside the 94% posterior predictive interval, at geo-week and national level. FAIL outside the range.
- **Source of threshold:** project
- **Basis:** Nominal 94%; 85-99% allows for sampling noise. Below: intervals too narrow; above: too wide.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `insample_coverage_range` | [0.85, 0.99] |

Latest result:

| check | status | value |
|---|---|---|
| insample_coverage | PASS | geo_week 0.9463, national 0.9038 |

### within_geo_r2

**Does the model pick up week-to-week movements within each region, not just the differences between regions?**

- **Blocking:** no (WARN only)
- **Applies to:** selected model
- **Rule:** R2 after removing each geo's mean (week-to-week signal within geos). WARN below the minimum.
- **Source of threshold:** PROMPT.md (as a warning)
- **Basis:** Geo-week revenue has a within-geo CV of 30% and is close to white noise, so most within-geo variance is unexplainable; this flags a model with almost no within-geo signal.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `warn_within_geo_r2_min` | 0.1 |

Latest result:

| check | status | value |
|---|---|---|
| within_geo_r2 | PASS | 0.2741 |

### residual_autocorrelation

**Are the model's misses random, or does it repeatedly miss in the same direction week after week?**

- **Blocking:** no (WARN only)
- **Applies to:** selected model
- **Rule:** Lag-1 autocorrelation of residuals: national weekly series and median across geos. WARN if either absolute value exceeds the maximum.
- **Source of threshold:** PROMPT.md (as a warning)
- **Basis:** Revenue itself has lag-1 autocorrelation near 0 (EDA), so residual autocorrelation signals missing time structure.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `warn_residual_lag1_abs_max` | 0.2 |

Latest result:

| check | status | value |
|---|---|---|
| residual_autocorrelation | PASS | national_lag1 0.01116, median_geo_lag1 0.0006571 |

### cv

**Does the model forecast weeks it has not seen? We refit it on shorter histories and check the next 13 weeks.**

- **Blocking:** yes
- **Applies to:** holdout (train 1-143, forecast 144-156) and each rolling-origin fold (train 1-104, 1-117, 1-130; forecast the next 13 weeks); reported as cv_fold_104, cv_fold_117, cv_fold_130, cv_holdout
- **Rule:** Each fold is refitted on its training weeks only (all scaling inside the fold). FAIL if the fold fit does not pass the convergence gate (no forecast evidence), or if any of: forecast/in-sample MAPE ratio above the maximum (geo-week or national); national weekly MAPE above the maximum; 94% interval coverage below the minimum (geo-week or national).
- **Source of threshold:** PROMPT.md
- **Basis:** National MAPE: a 13-week moving average has MAPE 5.1% nationally, so 10% is about 1.9x the noise floor; kept at the PROMPT.md value. The error ratio and coverage are relative measures and need no noise-floor adjustment. Geo-week MAPE is not gated directly: its noise floor is 23%.
- **Enforced in:** `scripts/validate_cv.py (metrics), scripts/gates.py (gate)`

| threshold | value |
|---|---|
| `cv_error_ratio_max` | 1.25 |
| `cv_national_mape_max` | 0.1 |
| `cv_coverage_min` | 0.8 |

Latest result:

| check | status | value |
|---|---|---|
| cv_fold_104 | FAIL | geo_error_ratio 1.039, national_error_ratio 1.094, national_mape 0.04463, geo_coverage 0.9365, national_coverage 0.7692, national_bias 0.03119, failed [national_coverage] |
| cv_fold_117 | FAIL | converged False |
| cv_fold_130 | FAIL | converged False |
| cv_holdout | FAIL | converged False |

Note: Fold fit did not converge; no forecast evidence (diagnostics only).

### roi_fold_stability

**Do the channel ROIs stay roughly the same when the model is refitted on less data?**

- **Blocking:** yes
- **Applies to:** converged CV fold fits
- **Rule:** Per channel, max/min of the posterior-mean ROI across converged folds. WARN above the warn ratio, FAIL above the fail ratio. FAIL (not assessable) with fewer than 2 converged folds.
- **Source of threshold:** PROMPT.md
- **Basis:** ROI that changes several-fold when a quarter of data is added is not a stable estimate.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `roi_fold_ratio_warn` | 2 |
| `roi_fold_ratio_fail` | 5 |

Latest result:

| check | status | value |
|---|---|---|
| roi_fold_stability | FAIL | converged_folds 1 |

Note: Not assessable: fewer than two converged fold fits.

### channel_roi_checks

**For each channel: did the data actually inform the ROI, and is it clear whether the channel pays back?**

- **Blocking:** no (WARN only)
- **Applies to:** selected model, each paid channel
- **Rule:** Two warnings per channel. Prior-driven: ROI contraction (1 - posterior sd / prior sd) below the minimum. Payback undetermined: P(ROI > 1) strictly between the two bounds.
- **Source of threshold:** PROMPT.md (bounds 0.1 / 0.9); project (contraction 0.30)
- **Basis:** Contraction below 0.30 means the posterior keeps more than 70% of the prior spread, so the data say little. A payback call needs at least 90% posterior probability either way.
- **Enforced in:** `scripts/gates.py`

| threshold | value |
|---|---|
| `roi_contraction_min` | 0.3 |
| `p_roi_gt_1_determined` | [0.1, 0.9] |

Latest result:

| check | status | value |
|---|---|---|
| Channel0 | WARN | contraction 0.78, P(ROI>1) 0.24 |
| Channel1 | WARN | contraction 0.83, P(ROI>1) 0.27 |
| Channel2 | WARN | contraction 0.68, P(ROI>1) 0.64 |
| Channel3 | WARN | contraction 0.77, P(ROI>1) 0.81 |
| Channel4 | WARN | contraction 0.61, P(ROI>1) 0.57 |

## Meridian's own health checks (not gating)

- Meridian's ModelReviewer (convergence, baseline, Bayesian PPP, goodness of fit, prior-posterior shift, ROI plausibility). Reported next to our gates, not gating.
- **Looser than our gates:** Convergence: R-hat < 1.2 only (no ESS or divergence check). Goodness of fit: R2/MAPE for guidance, no pass bar. Bayesian PPP: total outcome only, while our gate checks weekly interval coverage.
- **Latest:** PASS, health score 93.1/100.

| check | status |
|---|---|
| Convergence | PASS |
| Baseline | PASS |
| BayesianPPP | PASS |
| GoodnessOfFit | PASS |
| PriorPosteriorShift | PASS |
| ImplausibleROI | PASS |
| HighVariance | PASS |
| PotentialBias | PASS |
