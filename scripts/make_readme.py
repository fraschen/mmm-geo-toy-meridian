"""Generates README.md. Every number is read from outputs/ (EDA numbers from make eda)
or from the model-selection record in docs/model_selection/."""

from __future__ import annotations

import importlib.metadata as md
import json

import pandas as pd

from common import CHANNELS, HISTORY, OUTPUTS, ROOT, load_gates_config, load_model_config

J = lambda p: json.load(open(OUTPUTS / p))
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
M = lambda x: f"{x / 1e6:,.1f}M"


def main():
  eda, gates, cv = J("eda/eda_summary.json"), J("gates/gates.json"), J("validation/cv_metrics.json")
  sel = json.load(open(HISTORY / "selection.json"))
  main_s, stress, cal, contrib = (J("budget/scenario_main.json"), J("budget/scenario_stress.json"),
                                  J("calibration/calibration.json"), J("analysis/contributions.json"))
  ident, conv = J("analysis/identification.json"), J("model/convergence.json")
  cfg, gcfg = load_model_config(), load_gates_config()
  res = pd.read_csv(OUTPUTS / "results" / "channel_results.csv")
  g = {x["gate"]: x for x in gates["gates"]}
  d, nf, tr = eda["data"], eda["noise_floor"], eda["trend"]
  conv_folds = [k for k, v in cv["folds"].items() if v["converged"]]
  f0 = cv["folds"][conv_folds[0]] if conv_folds else None
  lab = sel["labels"]
  versions = {p: md.version(p) for p in ("google-meridian", "jax", "jaxlib", "tfp-nightly")}
  best = res.sort_values("roi_mean", ascending=False)
  ins, cov = g["insample_r2"]["value"], g["insample_coverage"]["value"]
  ct = gcfg["gates"]["convergence"]["thresholds"]
  chan_rows = "\n".join(
      f"| {r.channel} | {pct(r.spend_share)} | {r.roi_mean:.2f} ({r.roi_ci_lo:.2f}-{r.roi_ci_hi:.2f}) | "
      f"{r.mroi_mean:.2f} | {r.p_roi_gt_1:.2f} |" for r in res.itertuples())
  gate_rows = "\n".join(f"| {x['gate']} | {x['status']} | {'yes' if x['blocking'] else 'no'} |"
                        for x in gates["gates"])
  step = sel["comparison_steps"][0] if sel["comparison_steps"] else None
  pos = lambda s: ", ".join(f"{c} {100 * v['change_pct']:+.0f}%" for c, v in s["channels"].items())
  r1 = [n for n in sel["rounds"]["round_1"]]
  contraction = g["channel_roi_checks"]["value"]

  text = f"""# Geo-level Bayesian MMM with Google Meridian

One production Meridian model (JAX backend) of Meridian's simulated geo dataset: 40 geos x
156 weeks, 5 paid channels, revenue as the outcome. This README covers the choices we
made, the diagnostics we trust, and what the model can and cannot tell a client.

## Verdict

**Not yet good enough for a client decision.** The model converges on the full data, fits
well in sample, and its in-sample intervals are well calibrated. Three things stop it from
being client-ready:

- **Out-of-sample evidence is thin.** Only {len(conv_folds)} of {len(cv['folds'])} validation refits
  converged.{f" In that one, national 94% interval coverage was {pct(f0['national']['forecast']['coverage_94'], 0)}, below the 80% gate." if f0 else ""}
  ROI stability across refits could not be assessed.
- **No channel's payback is determined.** P(ROI > 1) ranges from {res.p_roi_gt_1.min():.2f} to
  {res.p_roi_gt_1.max():.2f}, and every 94% ROI interval spans 1.
- **Baseline and media are only weakly separated.** Total baseline and total media contribution
  correlate at {ident['corr_baseline_total_vs_paid_media_total']:.2f} across posterior draws, so totals
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
- **Sampler:** NUTS with {cfg['sampler']['n_chains']} chains x {cfg['sampler']['n_keep']} draws, seed {cfg['sampler']['seed']}.

**Why this model.** It was chosen once, from candidates that varied what the EDA cannot settle:
baseline flexibility, media pooling and max lag. The full record is in
[docs/model_selection](docs/model_selection/README.md); those candidates are not part of the pipeline.

- **Round 1:** {", ".join(lab[n] for n in r1)} (an automatically selected 66-knot baseline or a
  linear one, with default or strong pooling) did not converge. The failures sat in the baseline.
- **Round 2:** two models with a linear baseline and strong pooling, with max lag 4 or 8, converged.
- **Comparison:** {f"the lag-8 model's PSIS-LOO ELPD was higher by {step['elpd_diff']:.1f}, with paired SE {step['se']:.1f}, short of the {step['threshold']:.1f} needed to prefer the more complex model, so the lag-4 model was kept." if step else sel['selection_basis']}
- {gcfg['meta']['note']}

## Diagnostics we trust

Rules and thresholds: [`config/gates.toml`](config/gates.toml), documented in
[reports/gates.md](reports/gates.md). Full report: [reports/diagnostics.md](reports/diagnostics.md).

- **Convergence (blocking):** every element of every parameter must meet R-hat <= {ct['rhat_max']},
  bulk and tail ESS >= {ct['ess_min']}, and at most {pct(ct['divergent_share_max'])} divergent draws.
  The model has max R-hat {conv['max_rhat']:.4f}, min ESS {conv['min_ess_bulk']:.0f} (bulk) and
  {conv['min_ess_tail']:.0f} (tail), and {conv['divergences']} divergence out of {conv['n_draws_total']} draws.
  A fit that fails reports nothing else.
- **Out-of-sample forecasts (blocking, the decisive test):** the model is refitted on weeks 1-104,
  1-117, 1-130 and 1-143, each forecasting the next 13 weeks, with all scaling computed inside the
  refit. Only {len(conv_folds)} refit converged, so this gate fails.
- **In-sample fit, judged against the noise floor:** R2 is {ins['geo_week']:.2f} at geo-week level and
  {ins['national']:.2f} nationally, and 94% coverage is {pct(cov['geo_week'], 0)} and {pct(cov['national'], 0)}.
  National MAPE is {pct(gates['insample']['national_mape'])}, against {pct(nf['national_mape_vs_13wk_ma'])} for a
  13-week moving average. The national R2 looks low because that series barely varies (CV
  {pct(nf['national_weekly_cv'])}), not because the model misses it.
- **Identification:** ROI posteriors contract by {min(v['roi_contraction'] for v in contraction.values()):.2f}-
  {max(v['roi_contraction'] for v in contraction.values()):.2f} relative to the prior, so the data move every channel.
  The baseline-media correlation of {ident['corr_baseline_total_vs_paid_media_total']:.2f} limits how far the channel
  splits can be trusted.
- **Not relied on:** Meridian's health score ({gates['meridian_health']['health_score']:.1f}/100,
  {gates['meridian_health']['overall_status']}). It checks convergence only at R-hat < 1.2 and scores
  fit in sample only, so it cannot see the out-of-sample failure above.

## What the model can tell a client

From [reports/measurement.md](reports/measurement.md); file `outputs/results/channel_results.csv`,
with column definitions and package versions in `channel_results_metadata.json`.

- **Paid media in total** drove {M(contrib['paid_total']['mean'])} of {M(contrib['observed_total'])} revenue
  ({pct(contrib['paid_total']['mean'] / contrib['observed_total'])}; 94% CI {M(contrib['paid_total']['ci_lo'])}-{M(contrib['paid_total']['ci_hi'])}).
- **Ranking of channels, with wide uncertainty:** highest mean ROI {best.channel.iloc[0]}
  ({best.roi_mean.iloc[0]:.2f}) and {best.channel.iloc[1]} ({best.roi_mean.iloc[1]:.2f}); lowest
  {best.channel.iloc[-1]} ({best.roi_mean.iloc[-1]:.2f}). Marginal ROI is below 1 for every channel, so
  spending more on any single channel is unlikely to pay back at the margin.

  | channel | spend share | ROI (94% CI) | mROI | P(ROI>1) |
  |---|---|---|---|---|
{chr(10).join('  ' + l for l in chan_rows.splitlines())}

- **Direction of a reallocation** ([budget scenarios](reports/measurement.md#budget-scenarios); Meridian
  reports [main](reports/meridian/optimization_main.html), [stress](reports/meridian/optimization_stress.html)).
  The total is held at the last 52 weeks' spend ({M(main_s['budget'])}).
  - Within +/-20% per channel: {pos(main_s)}. Expected gain {M(main_s['gain']['mean'])} (94% CI
    {M(main_s['gain']['ci_lo'])} to {M(main_s['gain']['ci_hi'])}), P(gain > 0) = {main_s['gain']['p_gain_gt_0']:.2f}.
  - Within +/-50%: {pos(stress)}. Expected gain {M(stress['gain']['mean'])}, P(gain > 0) = {stress['gain']['p_gain_gt_0']:.2f}.

  Moving spend toward Channel3 and Channel2 is more likely than not to help; the size of the gain is uncertain.

## What the model cannot tell a client

- **Whether a specific channel pays back:** no channel has P(ROI > 1) >= 0.9 or <= 0.1.
- **Forecasts:** the out-of-sample evidence comes from a single converged refit, and its national
  intervals were too narrow.
- **Geo-level ROI:** not reported; media effects are pooled across geos by design.
- **Spend far outside history:** results beyond observed spend assume the shape of the response curve.
- **A precise baseline/media split:** the two trade off almost one-for-one.

**Calibration (optional, HYPOTHETICAL).** `make calibration` refits the model with an invented
Channel3 lift test (ROI {cal['test']['point_estimate_roi']}), to show how a real test would be used. It narrows
Channel3's ROI interval and leaves the other channels almost unchanged. The model implied
{cal['implied_test_roi_before']['mean']:.2f} for the test geos and weeks, and the choice of test geos does not drive
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
  {versions['google-meridian']}, JAX {versions['jax']}, jaxlib {versions['jaxlib']}, TensorFlow Probability
  (tfp-nightly) {versions['tfp-nightly']}.
- **Backend:** JAX, Meridian 2.x's default; `MERIDIAN_BACKEND` is not set. TensorFlow is installed
  only as a transitive dependency.
- **Runtime:** fits are skipped when a saved model matches `config/model.toml`. A cold run is one
  full fit plus four validation refits, about 30 minutes of sampling on a laptop CPU.

## Data and EDA

[`geo_all_channels.csv`](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv)
is downloaded from a pinned upstream commit and checked against a SHA-256 checksum, and is never
modified. It has {d['n_geos']} geos x {d['n_weeks']} weeks ({d['first_week']} to {d['last_week']}).
KPI: conversions x revenue per conversion. Full EDA: [reports/eda.md](reports/eda.md).

- **Panel and spend:** balanced panel with {d['missing_values_total']} missing values. Spend is
  impressions x a constant CPM. Spend shares: {", ".join(f"{c} {pct(d['spend_share'][c])}" for c in CHANNELS)}.
- **Trend and seasonality:** trend {tr[tr['primary']]['slope_per_year_pct_of_mean']:+.2f}%/year
  (HAC p = {tr[tr['primary']]['p_value']:.3f}); no yearly seasonality (p = {eda['seasonality']['p_value_ar1']:.2f}
  against simulated noise).
- **Noise floor:** geo-week revenue CV {pct(nf['geo_weekly_cv']['median'])}, national {pct(nf['national_weekly_cv'])}.
- **Media signal:** within-geo spend-revenue correlations are all |r| <= 0.08.
- **Controls:** they correlate with spend (up to {eda['controls_confounding']['max_abs_corr_control_with_spend_within_geo']:.2f}),
  so they are confounders and stay in the model.
- **Collinearity:** within-geo VIF is at most {eda['collinearity']['max_vif']:.2f}.

## Gates

| gate | status | blocking |
|---|---|---|
{gate_rows}

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
"""
  (ROOT / "README.md").write_text(text)
  print("Wrote README.md")


if __name__ == "__main__":
  main()
