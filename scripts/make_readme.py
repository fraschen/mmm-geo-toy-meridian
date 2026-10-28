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


def details(*items):
  return "\n<details><summary>Technical details</summary>\n\n" + "\n".join(f"- {i}" for i in items) + "\n\n</details>\n"


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

  text = f"""# Marketing mix model with Google Meridian

This project measures how much five paid media channels contribute to revenue, using
Google's Meridian marketing mix model on 3 years of weekly data from 40 regions (Meridian's
simulated dataset). It runs one model, in a container, from one command.

This page explains the choices we made, which checks we trust, and what the model can and
cannot tell a client.

## The short answer

**The model is useful for direction, not yet for decisions.** It fits the past well and its
uncertainty ranges are honest in-sample. But three things keep it from being client-ready:

- **We could not confirm it forecasts well.** We refit the model on shorter stretches of history
  to see how it predicts the weeks that follow. Only {len(conv_folds)} of those {len(cv['folds'])} refits
  ran reliably.{f" In that one, the forecast ranges were too narrow nationally: they captured {pct(f0['national']['forecast']['coverage_94'], 0)} of weeks, below our 80% bar." if f0 else ""}
- **No channel clearly pays for itself, or clearly doesn't.** The probability that a channel's ROI is
  above 1 ranges from {pct(res.p_roi_gt_1.min(), 0)} to {pct(res.p_roi_gt_1.max(), 0)}; we would need 90% (or 10%) to call it.
- **The model struggles to separate media from everything else.** It is confident about total revenue,
  less so about how much of it comes from media versus underlying demand.

Anything shared from this model should come with these caveats.
{details(f"Validation refits: train on weeks 1-104, 1-117, 1-130 and 1-143 (the last is the 13-week holdout), forecast the next 13 weeks; converged: {', '.join(k for k in conv_folds) or 'none'}." + (f" In the converged refit: geo-week forecast MAPE {pct(f0['geo_week']['forecast']['mape'])} (ratio to in-sample {f0['geo_week']['error_ratio']:.2f}, 94% coverage {pct(f0['geo_week']['forecast']['coverage_94'], 0)}); national MAPE {pct(f0['national']['forecast']['mape'])} (ratio {f0['national']['error_ratio']:.2f}, coverage {pct(f0['national']['forecast']['coverage_94'], 0)}, bias {100 * f0['national']['forecast']['mean_signed_pct_error']:+.1f}%)." if f0 else ""),
          f"P(ROI > 1) by channel: " + ", ".join(f"{r.channel} {r.p_roi_gt_1:.2f}" for r in res.itertuples()) + "; every 94% ROI interval spans 1.",
          f"Posterior correlation between total baseline and total paid-media contribution: {ident['corr_baseline_total_vs_paid_media_total']:.2f}.")}

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
{gcfg['meta']['note_plain']} Full record: [docs/model_selection](docs/model_selection/README.md).
{details("Spec: " + ", ".join(f"`{k} = {v}`" for k, v in cfg['model']['spec'].items()) + ".",
          "Priors (others Meridian defaults): " + ", ".join(f"`{k} ~ {v['dist']}(" + ", ".join(f"{a}={x}" for a, x in v.items() if a != 'dist') + ")`" for k, v in cfg['model']['priors'].items()) + ".",
          f"Sampler: {cfg['sampler']['n_chains']} chains, {cfg['sampler']['n_adapt']} adaptation + {cfg['sampler']['n_burnin']} burn-in + {cfg['sampler']['n_keep']} kept draws per chain, seed {cfg['sampler']['seed']}.",
          "Round 1 (" + ", ".join(lab[n] for n in r1) + "): AKS (66 knots) or linear baseline x default or strong media pooling; none converged (failures in the baseline parameters `knot_values`, `mu_t`, `tau_g`).",
          "Round 2 (R1, R2): linear baseline, strong pooling (`eta_m ~ HalfNormal(0.1)`), max lag 4 vs 8; both converged.",
          (f"PSIS-LOO: R2 vs R1 ELPD difference {step['elpd_diff']:+.1f} (paired SE {step['se']:.1f}); switching needs more than {step['threshold']:.1f}, so R1 is kept." if step else sel['selection_basis']),
          gcfg['meta']['note'])}

## The checks we trust

All rules are in [`config/gates.toml`](config/gates.toml) and explained in
[reports/gates.md](reports/gates.md); details in [reports/diagnostics.md](reports/diagnostics.md).

- **Did the computation run reliably?** {'Yes' if conv['converged'] else 'No'}. Meridian estimates the model by running {cfg['sampler']['n_chains']}
  independent simulations, which {'agree' if conv['converged'] else 'do not agree'} with each other. If they did not, we would report nothing.
- **Does it forecast weeks it has not seen?** This is the test that matters most, and the one the model
  does not pass yet (see above).
- **Does it fit the past?** {'Yes, within the limits of the data' if g['insample_r2']['status'] == 'PASS' and g['insample_coverage']['status'] == 'PASS' else 'Not well enough'}. Nationally it misses a typical week by
  {pct(gates['insample']['national_mape'])}, better than a simple 13-week average ({pct(nf['national_mape_vs_13wk_ma'])}).
  Its 94% ranges contain {pct(cov['geo_week'], 0)} of regional weeks, close to the 94% they should.
- **Did the data teach the model anything?** {'Yes. Every channel' if not any(v['prior_driven'] for v in contraction.values()) else 'Not for every channel. Some channels'}'s estimate moved away from the
  starting assumption{'' if not any(v['prior_driven'] for v in contraction.values()) else ' only a little'}.
- **What we don't rely on:** Meridian's built-in health score ({gates['meridian_health']['health_score']:.0f}/100). It only
  looks at how well the model fits the past, so it cannot see the forecasting problem.
{details(f"Convergence gate: every element of every parameter needs R-hat <= {ct['rhat_max']}, bulk and tail ESS >= {ct['ess_min']}, and at most {pct(ct['divergent_share_max'])} divergent draws. Model: max R-hat {conv['max_rhat']:.4f}, min ESS {conv['min_ess_bulk']:.0f} (bulk) / {conv['min_ess_tail']:.0f} (tail), {conv['divergences']} divergence(s) in {conv['n_draws_total']} draws.",
          f"In-sample R2: geo-week {ins['geo_week']:.2f}, national {ins['national']:.2f}; 94% coverage: geo-week {pct(cov['geo_week'])}, national {pct(cov['national'])}; MAPE: geo-week {pct(gates['insample']['geo_week_mape'])}, national {pct(gates['insample']['national_mape'])} (13-week moving average: {pct(nf['geo_mape_vs_13wk_ma']['median'])} and {pct(nf['national_mape_vs_13wk_ma'])}). The national R2 is modest because the national series barely varies (CV {pct(nf['national_weekly_cv'])}).",
          "ROI contraction vs prior (1 - posterior sd / prior sd): " + ", ".join(f"{c} {v['roi_contraction']:.2f}" for c, v in contraction.items()) + ".",
          f"Baseline-media posterior correlation {ident['corr_baseline_total_vs_paid_media_total']:.2f}.",
          f"Meridian health score {gates['meridian_health']['health_score']:.1f}/100 ({gates['meridian_health']['overall_status']}); its convergence check uses R-hat < 1.2 only.")}

## What the model can tell a client

Details in [reports/measurement.md](reports/measurement.md). ROI is revenue returned per unit
of spend: above 1, a channel pays for itself.

- **Media as a whole** drove about {M(contrib['paid_total']['mean'])} of {M(contrib['observed_total'])} revenue
  ({pct(contrib['paid_total']['mean'] / contrib['observed_total'], 0)}), somewhere between {M(contrib['paid_total']['ci_lo'])}
  and {M(contrib['paid_total']['ci_hi'])}.
- **Which channels look stronger:** {best.channel.iloc[0]} and {best.channel.iloc[1]} have the highest
  returns, and {best.channel.iloc[-1]} the lowest, but the ranges overlap a lot.

  | channel | share of spend | ROI (94% range) | marginal ROI | P(ROI > 1) |
  |---|---|---|---|---|
{chr(10).join('  ' + l for l in chan_rows.splitlines())}

  Marginal ROI (the return on the next unit of spend) is below 1 for every channel, so adding
  budget to any single channel is unlikely to pay back on its own.
- **Which way to move budget:** with the total held at last year's spend ({M(main_s['budget'])}), shifting
  up to 20% per channel toward Channel3 and Channel2 would add about {M(main_s['gain']['mean'])} of revenue
  a year, with a {pct(main_s['gain']['p_gain_gt_0'], 0)} chance of being an improvement. Shifting up to 50% would
  add about {M(stress['gain']['mean'])}, with a {pct(stress['gain']['p_gain_gt_0'], 0)} chance. The direction is fairly clear; the size
  is uncertain, and could even be a small loss
  ([scenarios](reports/measurement.md#budget-scenarios); Meridian's reports: [main](reports/meridian/optimization_main.html),
  [stress](reports/meridian/optimization_stress.html)).
{details(f"Paid media contribution: {M(contrib['paid_total']['mean'])} (94% CI {M(contrib['paid_total']['ci_lo'])}-{M(contrib['paid_total']['ci_hi'])}), {pct(contrib['paid_total']['mean'] / contrib['observed_total'])} of revenue.",
          f"Main scenario (+/-20%): {pos(main_s)}; gain {M(main_s['gain']['mean'])} (94% CI {M(main_s['gain']['ci_lo'])} to {M(main_s['gain']['ci_hi'])}), P(gain > 0) = {main_s['gain']['p_gain_gt_0']:.2f}.",
          f"Stress scenario (+/-50%): {pos(stress)}; gain {M(stress['gain']['mean'])} (94% CI {M(stress['gain']['ci_lo'])} to {M(stress['gain']['ci_hi'])}), P(gain > 0) = {stress['gain']['p_gain_gt_0']:.2f}.",
          f"Budget conserved to the optimizer's spend grid (difference {100 * main_s['budget_conserved']['relative_difference']:.3f}%). Beyond historical spend, results assume the response-curve shape.")}

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
{details(f"HYPOTHETICAL test: {cal['test']['channel']}, {cal['test']['start_date']} to {cal['test']['end_date']}, ROI {cal['test']['point_estimate_roi']} (SE {cal['test']['standard_error']}); calibrated prior mean {cal['prior_channel']['after_calibrated']['mean']:.2f}, sd {cal['prior_channel']['after_sd']:.2f}.",
          f"Model-implied ROI for the test regions and weeks before calibration: {cal['implied_test_roi_before']['mean']:.2f} (94% CI {cal['implied_test_roi_before']['ci_lo']:.2f}-{cal['implied_test_roi_before']['ci_hi']:.2f}).",
          f"Across {cal['geo_choice_check']['n_alternative_sets']} alternative test-region sets, implied ROI {cal['geo_choice_check']['implied_roi_mean_range'][0]:.2f}-{cal['geo_choice_check']['implied_roi_mean_range'][1]:.2f}: the choice of regions does not drive the result.")}

## Run it

```bash
make all      # the full pipeline: data checks, fit, reliability check, forecast tests,
              # Meridian reports, budget scenarios, gates, results and reports
make gates    # the acceptance checks only; fails if a blocking check fails
make docker   # build the container and check that it reproduces the analysis
docker run --rm mmm-geo-toy-meridian   # run the full pipeline in the container
```

Pinned environment (`requirements.lock`): Python 3.11, google-meridian {versions['google-meridian']},
JAX {versions['jax']}, jaxlib {versions['jaxlib']}, TensorFlow Probability (tfp-nightly)
{versions['tfp-nightly']}, on Meridian's default JAX backend. A cold run takes about 30 minutes;
saved models are reused when the configuration has not changed.

## The data

Meridian's simulated [geo dataset](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv):
{d['n_geos']} regions, {d['n_weeks']} weeks ({d['first_week']} to {d['last_week']}), five paid channels,
two controls, a promotion and one organic channel. It is downloaded from a fixed version and checked
before use. What we learned from it: [reports/eda.md](reports/eda.md).

- **Data quality:** the data are complete and clean.
- **Trend and seasonality:** revenue grows about {tr[tr['primary']]['slope_per_year_pct_of_mean']:.1f}% a year, with no
  meaningful yearly seasonality.
- **Noise:** weekly revenue is noisy, about {pct(nf['geo_weekly_cv']['median'], 0)} up or down per region and
  {pct(nf['national_weekly_cv'])} nationally, which limits how precise any model can be.
- **Media signal:** media's effect is hard to see in the raw data, so the model has to pool regions to find it.
{details(f"Balanced panel, {d['missing_values_total']} missing values; spend = impressions x a constant CPM. Spend shares: " + ", ".join(f"{c} {pct(d['spend_share'][c])}" for c in CHANNELS) + ".",
          f"Trend {tr[tr['primary']]['slope_per_year_pct_of_mean']:+.2f}%/year (HAC p = {tr[tr['primary']]['p_value']:.3f}); yearly breaks do not improve on a line (p = {tr['piecewise_yearly_vs_linear']['p_value']:.2f}).",
          f"Seasonality: p = {eda['seasonality']['p_value_ar1']:.2f} vs simulated noise; any yearly swing at most {pct(eda['seasonality']['bound']['peak_to_trough_upper_95'])} peak to trough (95% upper bound).",
          f"Within-region spend-revenue correlations all |r| <= 0.08; within-region VIF at most {eda['collinearity']['max_vif']:.2f}.",
          f"Controls correlate with spend up to {eda['controls_confounding']['max_abs_corr_control_with_spend_within_geo']:.2f}; their effects vary across regions (I2 {pct(eda['control_heterogeneity']['competitor_sales_control']['i2'], 0)} and {pct(eda['control_heterogeneity']['sentiment_score_control']['i2'], 0)}).")}

## Acceptance checks

| check | result | blocking |
|---|---|---|
{gate_rows}

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
"""
  (ROOT / "README.md").write_text(text)
  print("Wrote README.md")


if __name__ == "__main__":
  main()
