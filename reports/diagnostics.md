# Diagnostics: model health

Selected model: **linear_strong_pool_lag4** (R1: linear trend, `eta_m ~ HalfNormal(0.1)`, `max_lag = 4`), chosen by PSIS-LOO with the 2-SE simplicity rule.

Meridian's native reports: [health check](meridian/health_check.html), [Model Results Summary](meridian/results_summary.html).

## How the model was chosen
- Candidates differed on what the EDA cannot settle: baseline flexibility (knots), geo pooling of media effects (`eta_m`) and max lag. Specs, priors and protocol are in [candidates.md](candidates.md).
- Each fit had to pass a convergence gate before PSIS-LOO comparison. The fits ran in two rounds:
  - Round 1: C1-C3: knots (AKS vs linear) x media pooling (default vs strong); default sampler. C2 failed; C3 failed; C1 failed.
  - Round 2: R1-R2: after round 1 failed in the baseline parameters, linear knots and strong pooling (the docs' convergence checklist), with max lag 4 vs 8; default sampler. R1 passed; R2 passed.
- R2 vs R1: ELPD difference +3.6 (paired SE 2.5); switching needs more than 5.0, so R1 is kept.
- Selected: R1, PSIS-LOO ELPD -7449.5 (SE 54.9), 0 Pareto k values above 0.7.
- The candidate set and the convergence criteria were revised after earlier rounds failed to converge.

## Data and prior checks
- Data integrity: PASS (checksum, balanced 40 x 156 panel, no missing or duplicate rows, non-negative spend/impressions/KPI). Details in [eda.md](eda.md).
- Prior predictive: 100% of national weeks lie in the prior predictive 94% interval (PASS); the interval is wide (prior predictive 94% interval width / observed mean: 7.06), so this only rules out priors that cannot produce the data.

## Convergence
- Gate: R-hat <= 1.05, bulk and tail ESS >= 400, at most 20 divergent draws of 4000.
- Selected model: max R-hat 1.0091, min bulk ESS 441, min tail ESS 690, 1 divergence(s): PASS.

![](figures/diagnostics/convergence.png)

## In-sample fit and residuals
- R2: geo-week 0.767, national weekly 0.476; MAPE: geo-week 26.0%, national 4.0% (noise floor: a 13-week moving average has MAPE 23.1% per geo, 5.1% nationally).
- 94% interval coverage: geo-week 94.6%, national 90.4% (PASS). Within-geo R2 0.274 (PASS); residual lag-1 autocorrelation: national +0.011, median geo +0.001 (PASS).

![](figures/diagnostics/insample_fit.png)

![](figures/diagnostics/residuals.png)

## Holdout and rolling-origin CV
Folds train on weeks 1-104, 1-117, 1-130 and 1-143 (the last is the 13-week holdout) and forecast the next 13 weeks. Each fold is refitted on the truncated data, so all scaling is computed inside the fold. Forecasts use Meridian's equations with the linear baseline extended, conditional on observed media, controls, Promo and organic.

Findings:
- 1 of 4 fold fits passed the convergence gate; fold_117, fold_130, holdout did not, so they report diagnostics only.
- fold_104: geo-week forecast MAPE 26.6% vs in-sample 25.6% (ratio 1.04, gate <= 1.25); national MAPE 4.5% (gate <= 10%), ratio 1.09.
- fold_104: 94% interval coverage 94% at geo-week level but 77% nationally (gate >= 80%). The national forecast is biased by +3.1% on average and over-predicts in 54% of forecast weeks. Summing geos makes the national interval narrow relative to a shared bias, so a few percent of bias is enough to push national weeks outside it while geo-week intervals, dominated by geo-level noise, still cover.

| fold | train weeks | converged | geo MAPE | geo ratio | geo cov. | nat. MAPE | nat. ratio | nat. cov. | nat. bias | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fold_104 | 1-104 | yes | 26.6% | 1.04 | 94% | 4.5% | 1.09 | 77% | +3.1% | FAIL |
| fold_117 | 1-117 | no | - | - | - | - | - | - | - | FAIL |
| fold_130 | 1-130 | no | - | - | - | - | - | - | - | FAIL |
| holdout | 1-143 | no | - | - | - | - | - | - | - | FAIL |

![](figures/diagnostics/cv_forecast.png)

![](figures/diagnostics/roi_by_fold.png)

ROI stability across folds: FAIL (Not assessable: fewer than two converged fold fits.).

![](figures/diagnostics/calibration.png)

## Identification
- ROI contraction (1 - posterior sd / prior sd): Channel0 0.78, Channel1 0.83, Channel2 0.68, Channel3 0.77, Channel4 0.61 (warn below 0.3): the data move every channel's ROI away from the prior.
- Baseline vs media: across posterior draws, total baseline and total paid-media contribution have correlation -0.97. The model pins down their sum precisely but not the split between them; this is the main source of ROI uncertainty.

![](figures/diagnostics/roi_prior_posterior.png)

## Meridian health checks
Overall: PASS, health score 93.1/100. Where Meridian's thresholds are looser than our gates:

| check | status | comparison with our gates |
|---|---|---|
| Convergence | PASS | Meridian passes R-hat < 1.2; our gate uses R-hat <= 1.05 plus ESS and divergences |
| Baseline | PASS |  |
| BayesianPPP | PASS | Meridian checks only the total outcome; our gate checks weekly interval coverage |
| GoodnessOfFit | PASS | Meridian reports R2/MAPE for guidance only (no pass/fail bar); our in-sample and CV gates are explicit |
| PriorPosteriorShift | PASS |  |
| ImplausibleROI | PASS |  |
| HighVariance | PASS |  |
| PotentialBias | PASS |  |

## Gate summary (`make gates`)

Rules, thresholds and rationale: [gates.md](gates.md) (generated from `config/gates.toml`).

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

Blocking failures: cv_fold_104, cv_fold_117, cv_fold_130, cv_holdout, roi_fold_stability.

## Known limitations
- Out-of-sample evidence is thin: only one CV fold fit converged, and its national interval coverage misses the gate.
- Sampling is fragile: the selected model converges on the full data but not reliably on shorter windows.
- Baseline and media trade off almost one-for-one, so channel ROIs are wide and no channel's payback is determined (P(ROI > 1) between 0.1 and 0.9 for every channel).
- The linear-trend baseline cannot absorb week-level shocks common to all geos; the EDA found some common weekly variation beyond noise.
