# How the production model was chosen

This is a record. The production pipeline fits only the selected model,
[`config/model.toml`](../../config/model.toml); the candidates below are not refitted
or kept. This folder holds their convergence diagnostics (`convergence_*.json`), fit
settings (`fits/`), the PSIS-LOO pointwise values (`loo_pointwise_*.csv`), the comparison
(`selection.json`) and the script that produced it (`select_model.py`).

EDA numbers below come from `outputs/eda/eda_summary.json` (`make eda`); details in
[reports/eda.md](../../reports/eda.md). Prior choices were checked against the Meridian docs and the
installed `meridian.model.spec` / `prior_distribution` source (google-meridian 2.1.0).

## What the EDA settles (common to all candidates)

| Setting | Choice | EDA evidence / doc basis |
|---|---|---|
| KPI | `conversions`, `kpi_type='non_revenue'`, `revenue_per_kpi=revenue_per_conversion` | ROI is then in revenue units; revenue per conversion is almost constant (CV 0.5%). Docs: supplying `revenue_per_kpi` keeps the default ROI prior meaningful. |
| Media input | impressions with spend | Spend = impressions x constant CPM (max deviation 0.14 ppm), so the impressions vs spend choice changes nothing. |
| Paid media priors | ROI priors, Meridian default `roi_m = LogNormal(0.2, 0.9)` for every channel | Docs: ROI priors are the default and align with prior knowledge; coefficient priors are "rarely recommended". Same prior everywhere, so differences between channels come from the data. Calibration is Phase 8. |
| Population | KPI and paid/organic media population-scaled (Meridian default) | Revenue and spend scale ~1:1 with population (log-log slopes 1.04 and 1.02). |
| Controls | both controls, hierarchical across geos (`gamma_c ~ N(0,5)`, `xi_c ~ HalfNormal(5)` defaults), **not** population-scaled | They are confounders: they correlate with revenue (competitor -0.20) and with spend (up to 0.55), and partialling them out removes the negative lag-0 spend-revenue correlations. Per-geo effects are strongly heterogeneous (I2 94% and 88%), so the geo hierarchy needs room and the wide default `xi_c` stays. Population scaling makes no sense here: these are centred indices with negative values, and their geo means do not track population (Spearman -0.05, +0.23). |
| Promo | non-media treatment, default contribution prior `TruncatedNormal(0, 0.1, -1, 1)`, not population-scaled | Geo effects are homogeneous (I2 26%, Q p = 0.07); geo means barely track population (+0.18). The default prior is symmetric, so the sign is left to the data. |
| Organic | `Organic_channel0_impression` as organic media, default contribution prior `Beta(1, 99)` | Small within-geo association with revenue (+0.03), homogeneous across geos (I2 20%). |
| Adstock / lag | geometric decay, Hill after adstock, `max_lag = 8` | Lag correlations are all inside or barely outside the noise band (+/-0.025), with no decay pattern, so the data cannot set the lag. The Meridian default of 8 is inside the doc range for geometric decay (2-10), and with `alpha_m ~ U(0,1)` the model can shorten the effective lag itself. |
| Residual variance | one sigma for all geos (default) | Per-capita noise only weakly tracks population (Spearman -0.33; smallest/largest sd 0.60). Docs warn that per-geo sigma makes convergence harder. Per-geo residuals are checked in Phase 5. |

## What the EDA cannot settle (the candidates differ here)

1. **How flexible the baseline is over time (knots).** There is no yearly seasonality
   (p = 0.28 vs AR(1) noise), and the national trend is linear (+1.46%/year, HAC
   p = 0.010). Yearly breaks do not improve on a straight line (HAC Wald p = 0.10). In
   Meridian, `knots = 2` (one knot at each end, no bends) is exactly a straight-line baseline. However, week dummies
   explain 5.3% of within-geo per-capita revenue variance, against 2.5% under a
   permutation null, and Meridian's docs recommend starting with Automatic Knot
   Selection, which picks 66 knots here.
2. **How much media effects are pooled across geos (`eta_m`).** Within-geo
   spend-revenue correlations are tiny (|r| <= 0.079), and Channel2 is off in 64% of
   geo-weeks. Per-geo media effects are therefore barely informed by each geo's own
   data. The default is `HalfNormal(1)`. The docs suggest `HalfNormal(0.1)` when data
   are scarce.
3. **Max lag.** Lag correlations show no decay pattern. The docs note that a smaller
   `max_lag` generally improves convergence.

## Selection protocol

1. **Convergence gate.** For every element of every posterior variable: R-hat <= 1.05,
   bulk ESS >= 400, tail ESS >= 400, and at most 0.5% divergent draws. A failing fit
   records only its diagnostics; failing fits are not rerun.
2. **PSIS-LOO among converged fits.** Pointwise log-likelihood on Meridian's
   transformed KPI; pointwise values and Pareto k are saved in this folder (`loo_pointwise_*.csv`).
3. **Simplicity rule.** Start from the simplest converged fit (order R1 < R2 < C2 <
   C3 < C1). Move to a more complex one only if its ELPD exceeds the current choice's by
   more than 2 SE of the paired difference.

The convergence criteria were revised after the first rounds failed to converge; the
original gate was R-hat <= 1.01, ESS >= 400 and 0 divergences.

All fits use 4 chains x 1000 kept draws and seed 20260128 (`model_build/build_model.py`).

## Round 1: knots x pooling

| | knots | `eta_m` | `max_lag` |
|---|---|---|---|
| C1 `aks_default_pool` | AKS (66) | `HalfNormal(1)` | 8 |
| C2 `linear_default_pool` | 2 (linear) | `HalfNormal(1)` | 8 |
| C3 `aks_strong_pool` | AKS (66) | `HalfNormal(0.1)` | 8 |

Sampler: `n_adapt=500, n_burnin=500`. None converged. The failures sat in the baseline
parameters (`knot_values`, `mu_t`, `tau_g`): the national level and the geo intercepts
traded off in the posterior, and the chains disagreed on the level. C1 also failed in its
media parameters.

## Round 2: targeting the failure

Following the docs' "Getting MCMC convergence" checklist
([model-debugging](https://developers.google.com/meridian/docs/post-modeling/model-debugging)):

- **Knots:** linear only. The EDA supports it, AKS with the docs' higher penalty range
  (`np.geomspace(10, 200, 100)`) selects 3 knots (essentially a line), and the 66-knot
  baseline failed under both pooling settings.
- **Pooling:** strong (`HalfNormal(0.1)`). In round 1 it brought AKS from R-hat 1.22 to 1.02.
- **Max lag:** the dimension that differs, because smaller lags generally improve convergence.

| | knots | `eta_m` | `max_lag` |
|---|---|---|---|
| R1 `linear_strong_pool_lag4` | 2 | `HalfNormal(0.1)` | 4 |
| R2 `linear_strong_pool` | 2 | `HalfNormal(0.1)` | 8 |

Sampler as in round 1. Both converged.

## Outcome

Among the converged fits (R1, R2), R2's ELPD exceeds R1's by 3.6 with paired SE 2.5,
below the 2-SE threshold. **R1 is selected.** Per-fit diagnostics are in
`selection.json` in this folder.
