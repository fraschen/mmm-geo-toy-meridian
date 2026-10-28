# Candidate specifications and selection protocol (fixed before fitting)

EDA numbers below come from `outputs/eda/eda_summary.json` (`make eda`); details in
[eda.md](eda.md). Prior choices were checked against the Meridian docs and the
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

1. **How flexible the baseline is over time (knots).** Two readings of the EDA conflict.
   - *Smooth:* there is no yearly seasonality (p = 0.28 vs AR(1) noise), the national
     trend is linear (+1.46%/year, HAC p = 0.010), and yearly breaks do not improve on a
     straight line (HAC Wald p = 0.10, borderline). In Meridian, `knots = 2` puts knots at
     weeks 0 and 155 and interpolates linearly between them (checked in
     `meridian.model.knots.get_knot_info`), so it is exactly a linear-trend baseline: the
     smallest knot count the EDA supports.
   - *Flexible:* week dummies explain 5.3% of within-geo per-capita revenue variance,
     against 2.5% under a permutation null. Some of that common weekly variation may be
     baseline shocks that a straight line would push into media (18-30% of within-geo
     spend variance is common national timing). Meridian's docs recommend starting with
     Automatic Knot Selection, which picks 66 knots on this data.
   The EDA cannot separate baseline shocks from nationally timed media and control
   effects, so both ends are candidates. AKS learns from the data, so in Phase 4 CV it is
   rerun inside each training fold.
2. **How much media effects are pooled across geos (`eta_m`).** Within-geo spend-revenue
   correlations are tiny (|r| <= 0.079), and Channel2 is off in 64% of geo-weeks. Per-geo
   media effects are therefore barely informed by each geo's own data. Meridian's default
   `eta_m ~ HalfNormal(1)` allows large geo spread. The docs recommend
   `HalfNormal(0.1)` when data are scarce. The EDA cannot measure geo heterogeneity of
   media effects directly.

| | C1 `aks_default_pool` | C2 `linear_default_pool` | C3 `aks_strong_pool` |
|---|---|---|---|
| knots | AKS (`knots.AKS(data).automatic_knot_selection()`, 66 knots on the full data) | 2 (linear trend) | AKS (66) |
| `eta_m` (geo spread of media effects) | `HalfNormal(1)` (default) | `HalfNormal(1)` (default) | `HalfNormal(0.1)` |
| `media_effects_dist` | `log_normal` (default) | `log_normal` | `log_normal` |
| everything else | as in the table above | same | same |
| complexity rank | 3 (most) | 1 (least: 64 fewer time parameters) | 2 (C1 with a tighter pooling prior) |

## Sampler (identical for all candidates; not changed after seeing results)

`sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=20260128)`,
NUTS defaults otherwise (`max_tree_depth=10`). Each fit is saved as
`model_build/<candidate>/meridian_model.binpb` with `meridian_serde.save_meridian`.

## Selection protocol

1. **Convergence gate (per candidate).** For every element of every variable in the
   posterior: R-hat <= 1.01, bulk ESS >= 400, tail ESS >= 400, and 0 divergent transitions.
   A candidate that fails any check is dropped, and only its convergence diagnostics are
   recorded: no ROI, fit or contributions anywhere. No reruns with more samples.
2. **PSIS-LOO among converged candidates.** The pointwise log-likelihood is computed from
   the posterior draws on Meridian's likelihood scale (`Normal(y_gt | mu_gt, sigma)` on the
   transformed KPI, which is identical across candidates). ELPD is estimated with
   ArviZ PSIS-LOO. Pointwise values and Pareto-k go to `outputs/selection/`.
3. **Simplicity rule.** Start from the simplest converged candidate, in the order
   C2 < C3 < C1. Move to the next more complex converged candidate only if its ELPD
   exceeds the current choice's by more than 2 SE of the paired difference. Steps:
   C2 vs C3, then the current choice vs C1.
4. If only one candidate converges, it is selected, and the reports say plainly that it
   won by convergence, not by fit.

## Round 2 (after round 1 failed the convergence gate)

No round-1 candidate converged. C2 and C3 failed narrowly, in the baseline
parameters only (`mu_t`, `knot_values`, `tau_g`). C1 failed broadly (max R-hat 1.22,
39 divergences). Per the protocol, all three were dropped and none was rerun. Their
convergence diagnostics are kept in `outputs/selection/convergence_*.json`. Round 2
was chosen **after** seeing round 1 fail, following the docs' "Getting MCMC
convergence" checklist ([model-debugging](https://developers.google.com/meridian/docs/post-modeling/model-debugging)):

- **Knots:** a straight line. The EDA does not reject it (p = 0.10). AKS with the docs'
  higher penalty range (`np.geomspace(10, 200, 100)`) selects 3 knots (weeks 0, 2, 155),
  essentially a line. The 66-knot AKS baseline failed under both pooling settings.
- **Pooling:** `eta_m ~ HalfNormal(0.1)`. This is the docs' suggestion when the geo
  hierarchy is hard to identify, and in round 1 it brought AKS from R-hat 1.22 to 1.02.
- **Max lag:** the dimension that differs. The EDA cannot set it, and the docs note that
  smaller `max_lag` generally improves convergence.

| | knots | `eta_m` | `max_lag` | complexity rank |
|---|---|---|---|---|
| R1 `linear_strong_pool_lag4` | 2 | `HalfNormal(0.1)` | 4 | 1 |
| R2 `linear_strong_pool` | 2 | `HalfNormal(0.1)` | 8 | 2 |

Everything else, the sampler, and the protocol (convergence gate, no reruns,
PSIS-LOO with the 2-SE rule, R1 < R2) are unchanged.
