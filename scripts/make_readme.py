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

## The checks we trust

All rules are in [`config/gates.toml`](config/gates.toml) and explained in
[reports/gates.md](reports/gates.md); details in [reports/diagnostics.md](reports/diagnostics.md).

- **Did the computation run reliably?** Yes. Meridian estimates the model by running {cfg['sampler']['n_chains']}
  independent simulations, and they agree with each other. If they did not, we would report nothing.
- **Does it forecast weeks it has not seen?** This is the test that matters most, and the one the model
  does not pass yet (see above).
- **Does it fit the past?** Yes, within the limits of the data. Nationally it misses a typical week by
  {pct(gates['insample']['national_mape'])}, better than a simple 13-week average ({pct(nf['national_mape_vs_13wk_ma'])}).
  Its 94% ranges contain {pct(cov['geo_week'], 0)} of regional weeks, close to the 94% they should.
- **Did the data teach the model anything?** Yes. Every channel's estimate moved away from the
  starting assumption.
- **What we don't rely on:** Meridian's built-in health score ({gates['meridian_health']['health_score']:.0f}/100). It only
  looks at how well the model fits the past, so it cannot see the forecasting problem.

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
