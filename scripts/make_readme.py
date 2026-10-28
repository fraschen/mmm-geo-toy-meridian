"""Generates README.md. Every number is read from outputs/ (EDA numbers from make eda)."""

from __future__ import annotations

import importlib.metadata as md
import json

import pandas as pd

from common import CHANNELS, OUTPUTS, ROOT

J = lambda p: json.load(open(OUTPUTS / p))
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
M = lambda x: f"{x / 1e6:,.1f}M"


def main():
  eda, sel, gates, cv = (J("eda/eda_summary.json"), J("selection/selection.json"),
                         J("gates/gates.json"), J("validation/cv_metrics.json"))
  main_s, stress, cal, contrib = (J("budget/scenario_main.json"), J("budget/scenario_stress.json"),
                                  J("calibration/calibration.json"), J("analysis/contributions.json"))
  ident = J("analysis/identification.json")
  res = pd.read_csv(OUTPUTS / "results" / "channel_results.csv")
  g = {x["gate"]: x for x in gates["gates"]}
  d, nf, te, tr = eda["data"], eda["noise_floor"], eda["time_effects"], eda["trend"]
  conv_folds = [k for k, v in cv["folds"].items() if v["converged"]]
  f0 = cv["folds"][conv_folds[0]] if conv_folds else None
  lab = sel["labels"]
  versions = {p: md.version(p) for p in ("google-meridian", "jax", "jaxlib", "tfp-nightly")}
  best = res.sort_values("roi_mean", ascending=False)
  ins = g["insample_r2"]["value"]
  chan_rows = "\n".join(
      f"| {r.channel} | {pct(r.spend_share)} | {r.roi_mean:.2f} ({r.roi_ci_lo:.2f}-{r.roi_ci_hi:.2f}) | "
      f"{r.mroi_mean:.2f} | {r.p_roi_gt_1:.2f} |" for r in res.itertuples())
  gate_rows = "\n".join(f"| {x['gate']} | {x['status']} | {'yes' if x['blocking'] else 'no'} |"
                        for x in gates["gates"])
  steps = "; ".join(f"{lab[s['challenger']]} vs {lab[s['current']]}: ELPD {s['elpd_diff']:+.1f} "
                    f"(SE {s['se']:.1f}, needs > {s['threshold']:.1f})" for s in sel["comparison_steps"])
  rounds = "\n".join(
      f"- **{r.replace('_', ' ').capitalize()}.** {sel['round_notes'][r]} Result: "
      + ", ".join(f"{lab[n]} {'converged' if sel['converged'][n] else 'did not converge'}" for n in names) + "."
      for r, names in sel["rounds"].items())
  pos = lambda s: ", ".join(f"{c} {100 * v['change_pct']:+.0f}%" for c, v in s["channels"].items())

  text = f"""# Geo-level Bayesian MMM with Google Meridian

A reproducible Meridian (JAX backend) marketing mix model of Meridian's simulated geo
dataset: 40 geos x 156 weeks, 5 paid channels, revenue as the outcome.

## Verdict

**Not yet good enough for a client decision.** The selected model ({lab[sel['selected']]}: a
linear-trend baseline with strongly pooled media effects and a 4-week max lag) converges on the full
data, fits well in sample, and its intervals are well calibrated in sample. Three things stop it
from being client-ready:

- **Out-of-sample evidence is thin.** Only {len(conv_folds)} of {len(cv['folds'])} rolling-origin CV
  fold fits converged.{f" In that fold, national 94% interval coverage was {pct(f0['national']['forecast']['coverage_94'], 0)}, below the 80% gate." if f0 else ""}
  ROI stability across folds could not be assessed.
- **No channel's payback is determined.** P(ROI > 1) ranges from {res.p_roi_gt_1.min():.2f} to
  {res.p_roi_gt_1.max():.2f}, and every 94% ROI interval spans 1.
- **Baseline and media are only weakly separated.** Total baseline and total media contribution
  correlate at {ident['corr_baseline_total_vs_paid_media_total']:.2f} across posterior draws.

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
{versions['google-meridian']}, JAX {versions['jax']}, jaxlib {versions['jaxlib']}, TensorFlow Probability
(tfp-nightly) {versions['tfp-nightly']}. Backend: JAX (Meridian 2.x default; `MERIDIAN_BACKEND` is not set).
TensorFlow is installed only as a transitive dependency. Long runs are wrapped in `caffeinate` on
macOS. A full `make all` on a laptop CPU takes a few hours, mostly MCMC; saved models are reused.

## Data

[`geo_all_channels.csv`](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv),
downloaded by `scripts/download_data.py` from a pinned upstream commit and checked against a SHA-256
checksum. The raw file is read-only and never modified. It has {d['n_geos']} geos x {d['n_weeks']} weeks
({d['first_week']} to {d['last_week']}). KPI: conversions, with revenue per conversion, so outcomes
are in revenue. Paid media: Channel0-4 impressions and spend. Controls: competitor sales,
sentiment. Promo is a non-media treatment and Organic_channel0 is organic media.

## EDA

Full report: [reports/eda.md](reports/eda.md); Meridian's EDA:
[model_build/eda_default/eda.html](model_build/eda_default/eda.html).

- **Data quality:** the panel is balanced, with {d['missing_values_total']} missing values. Spend is
  impressions x a constant CPM in every channel. Spend shares:
  {", ".join(f"{c} {pct(d['spend_share'][c])}" for c in CHANNELS)}.
- **Trend and seasonality:** a national trend of {tr[tr['primary']]['slope_per_year_pct_of_mean']:+.2f}%/year
  (HAC p = {tr[tr['primary']]['p_value']:.3f}), which a straight line describes (yearly breaks
  p = {tr['piecewise_yearly_vs_linear']['p_value']:.2f}). No yearly seasonality
  (p = {eda['seasonality']['p_value_ar1']:.2f} vs simulated noise).
- **Noise floor:** geo-week revenue CV {pct(nf['geo_weekly_cv']['median'])}; national CV
  {pct(nf['national_weekly_cv'])}; a 13-week moving average misses national revenue by
  {pct(nf['national_mape_vs_13wk_ma'])}. Lag-1 autocorrelation is about 0.
- **Weak media signal:** within-geo spend-revenue correlations are all |r| <= 0.08, with no clear lag
  decay. The controls confound spend and revenue (they correlate with spend up to
  {eda['controls_confounding']['max_abs_corr_control_with_spend_within_geo']:.2f}), and their
  effects vary strongly across geos (I2 {pct(eda['control_heterogeneity']['competitor_sales_control']['i2'], 0)}
  and {pct(eda['control_heterogeneity']['sentiment_score_control']['i2'], 0)}).
- **Collinearity is low:** within-geo VIF is at most {eda['collinearity']['max_vif']:.2f}. Revenue and
  spend scale about 1:1 with population.

## Candidates and selection

Details: [reports/candidates.md](reports/candidates.md). Convergence gate: R-hat <= 1.05,
bulk/tail ESS >= 400, <= 0.5% divergent draws.

{rounds}

Among the converged fits, PSIS-LOO with a 2-SE simplicity rule ({steps}) keeps
**{lab[sel['selected']]}** (`{sel['selected']}`).

## Results

- **Diagnostics** ([reports/diagnostics.md](reports/diagnostics.md)): the selected model converges
  (max R-hat {g['convergence']['value']['max_rhat']:.4f}). In-sample R2 is {ins['geo_week']:.2f}
  (geo-week) and {ins['national']:.2f} (national); 94% coverage is
  {pct(g['insample_coverage']['value']['geo_week'], 0)} and {pct(g['insample_coverage']['value']['national'], 0)}.
  Out-of-sample checks fail as described in the verdict. Meridian's own health check scores
  {gates['meridian_health']['health_score']:.1f}/100 ({gates['meridian_health']['overall_status']}), using looser
  thresholds than our gates.
- **Measurement** ([reports/measurement.md](reports/measurement.md)): paid media drove
  {M(contrib['paid_total']['mean'])} of the {M(contrib['observed_total'])} revenue
  ({pct(contrib['paid_total']['mean'] / contrib['observed_total'])}; 94% CI
  {M(contrib['paid_total']['ci_lo'])}-{M(contrib['paid_total']['ci_hi'])}). Highest mean ROI:
  {best.channel.iloc[0]} {best.roi_mean.iloc[0]:.2f} and {best.channel.iloc[1]} {best.roi_mean.iloc[1]:.2f};
  lowest: {best.channel.iloc[-1]} {best.roi_mean.iloc[-1]:.2f}. Marginal ROI is below 1 for every channel.

  | channel | spend share | ROI (94% CI) | mROI | P(ROI>1) |
  |---|---|---|---|---|
{chr(10).join('  ' + l for l in chan_rows.splitlines())}

- **Budget recommendations** ([reports/measurement.md](reports/measurement.md#budget-scenarios);
  Meridian reports [main](reports/meridian/optimization_main.html),
  [stress](reports/meridian/optimization_stress.html)): the total is held at the last 52 weeks'
  spend ({M(main_s['budget'])}).
  - Within +/-20% per channel: {pos(main_s)}. Expected gain {M(main_s['gain']['mean'])} (94% CI
    {M(main_s['gain']['ci_lo'])} to {M(main_s['gain']['ci_hi'])}), P(gain > 0) = {main_s['gain']['p_gain_gt_0']:.2f}.
  - Within +/-50%: {pos(stress)}. Expected gain {M(stress['gain']['mean'])}, P(gain > 0) =
    {stress['gain']['p_gain_gt_0']:.2f}.

  Moving spend toward Channel3 and Channel2 is more likely than not to help, but how much is uncertain.
  Beyond historically observed spend, the results assume the response-curve shape.
- **Calibration (HYPOTHETICAL lift test):** a Channel3 test reporting ROI
  {cal['test']['point_estimate_roi']} narrows Channel3's ROI interval and leaves the other channels
  almost unchanged. The model implied {cal['implied_test_roi_before']['mean']:.2f} for the test geos and
  weeks, and the choice of test geos does not drive the result.

Channel-level file: `outputs/results/channel_results.csv`, with column definitions and package
versions in `channel_results_metadata.json`.

## Gates

`make gates` exits non-zero on any blocking failure. Rules, thresholds and their rationale live in
[`config/gates.toml`](config/gates.toml), which the scripts read. [reports/gates.md](reports/gates.md)
documents every gate (rule, threshold, source, basis, latest value) and is generated from that file.

| gate | status | blocking |
|---|---|---|
{gate_rows}

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
config/gates.toml         gate rules and thresholds (single source)
reports/                  eda.md, candidates.md, diagnostics.md, measurement.md, gates.md, figures/,
                          meridian/ (Meridian HTML reports)
Makefile, Dockerfile, requirements.lock
```
"""
  (ROOT / "README.md").write_text(text)
  print("Wrote README.md")


if __name__ == "__main__":
  main()
