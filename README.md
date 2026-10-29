# Marketing mix model with Google Meridian

This project measures how much five paid media channels contribute to revenue, using
Google's Meridian marketing mix model on 3 years of weekly data from 40 geos (Meridian's
simulated dataset). It runs one model, in a container, from one command.

This page explains the choices we made, which checks we trust, and what the model can and
cannot tell a client. Numbers here are from the last full run; the full set is in
[reports/results.md](reports/results.md) and `outputs/`.

## The short answer

**The model is useful for direction, not yet for decisions.** It fits the past well and its
uncertainty ranges are honest in-sample. But three things keep it from being client-ready:

- **We could not confirm it forecasts well.** We refit the model on shorter stretches of history
  to see how it predicts the weeks that follow. Only 1 of those 4 refits ran reliably. In that one,
  the forecast ranges were too narrow nationally: they captured 77% of weeks, below our 80% bar.
- **No channel clearly pays for itself, or clearly doesn't.** The probability that a channel's ROI is
  above 1 ranges from 24% to 81%; we would need 90% (or 10%) to call it.
- **The model struggles to separate media from everything else.** It is confident about total revenue,
  less so about how much of it comes from media versus underlying demand.

Anything shared from this model should come with these caveats.

## The model

Defined in [`config/model.toml`](config/model.toml):

- **Underlying demand** follows a straight-line trend (`knots = 2`: one knot at each end of the
  series, so no bends), because the data show steady growth and no yearly seasonality.
- **Media** has effects that carry over for up to 4 weeks and that flatten out as spend grows.
- **Geos share information:** a channel is assumed to work similarly across geos, because
  each geo's own data are too noisy to estimate it alone.
- **Every channel starts from the same prior belief** about its return, so differences between
  channels come from the data.
- **Competitor sales and consumer sentiment** are included, because they move with both our spend
  and our revenue. Leaving them out would distort media's effect.

**Why this model.** Its choices follow from what the EDA could and could not settle, and from what
ran reliably on this data in earlier attempts: a more flexible baseline (Meridian's automatic knot
selection) and looser pooling across geos did not sample reliably, so we kept the baseline a straight
line and pooled media effects strongly.

## The checks we trust

The rules are in [`config/gates.toml`](config/gates.toml) and explained in
[reports/results.md](reports/results.md#part-3-acceptance-gates); the diagnostics are in
[Part 1](reports/results.md#part-1-diagnostics).

- **Did the computation run reliably?** Yes. Meridian estimates the model by running 4 independent
  simulations, and they agree with each other. If they did not, we would report nothing.
  The candidate set and the convergence criteria were revised after earlier rounds failed to
  converge; the original convergence gate was R-hat <= 1.01, bulk/tail ESS >= 400 and 0 divergences.
- **Does it forecast weeks it has not seen?** This is the test that matters most, and the one the
  model does not pass yet (see above).
- **Does it fit the past?** Yes, within the limits of the data. Nationally it misses a typical week by
  4.0%, better than a simple 13-week average (5.1%). Its 94% ranges contain 95% of geo-level weeks,
  close to the 94% they should.
- **Did the data teach the model anything?** Yes. Every channel's estimate moved away from the
  starting assumption.
- **What we don't rely on:** Meridian's built-in health score (93/100). It only looks at how well the
  model fits the past, so it cannot see the forecasting problem.

Status at the last run: the data, prior, convergence and in-sample checks pass; the four forecast
checks and the ROI-stability check fail; each channel carries a warning that its payback is undetermined.

## What the model can tell a client

Details in [reports/results.md](reports/results.md#part-2-measurement); the channel table is also in
`outputs/05_report/channel_results.csv`.

- **Media as a whole** drove about 256M of 1,319M revenue (19%), somewhere between 177M and 364M.
- **Which channels look stronger:** Channel3 and Channel4 have the highest returns, and Channel0 the
  lowest, but the ranges overlap a lot.

  | channel | share of spend | ROI (94% range) | marginal ROI | P(ROI > 1) |
  |---|---|---|---|---|
  | Channel0 | 18.5% | 0.76 (0.18-1.81) | 0.34 | 0.24 |
  | Channel1 | 14.3% | 0.82 (0.25-1.67) | 0.43 | 0.27 |
  | Channel2 | 5.5% | 1.27 (0.39-2.54) | 0.69 | 0.64 |
  | Channel3 | 40.0% | 1.40 (0.60-2.34) | 0.76 | 0.81 |
  | Channel4 | 21.7% | 1.28 (0.27-3.24) | 0.49 | 0.57 |

  Marginal ROI (the return on the next unit of spend) is below 1 for every channel, so adding
  budget to any single channel is unlikely to pay back on its own.
- **Which way to move budget:** with the total held at last year's spend (71.4M), shifting up to 20%
  per channel toward Channel3 and Channel2 would add about 1.8M of revenue a year, with an 84% chance
  of being an improvement. Shifting up to 50% would add about 2.5M, with a 77% chance. The direction is
  fairly clear; the size is uncertain, and could even be a small loss
  ([scenarios](reports/results.md#budget-scenarios); Meridian's reports:
  [main](reports/meridian/optimization_main.html), [stress](reports/meridian/optimization_stress.html)).

## What the model cannot tell a client

- **Whether any specific channel pays back.** The ranges are too wide.
- **Reliable forecasts.** Not yet tested well enough.
- **Results by geo.** The model shares information across geos, so it reports national results only.
- **What happens at much higher spend than ever before.** Beyond past levels, results depend on
  assumptions about how returns flatten.
- **A precise split between media and underlying demand.**

**Using a real experiment (optional, illustrative).** `make calibration` shows how the result
of a lift test would sharpen the model. We used an invented (HYPOTHETICAL) Channel3 test: it
narrowed Channel3's range and left the others unchanged, and the choice of test geos did not drive
the result. It is not part of the production model.

## Run it

```bash
make all      # the full pipeline: data checks, fit, reliability check, forecast tests,
              # Meridian reports, budget scenarios, gates, results and reports
make gates    # the acceptance checks only; fails if a blocking check fails
make docker   # build the container and check that it reproduces the analysis
docker run --rm mmm-geo-toy-meridian   # run the full pipeline in the container
```

Pinned environment (`requirements.lock`): Python 3.11, google-meridian 2.1.0, JAX 0.10.2,
jaxlib 0.10.2, TensorFlow Probability (tfp-nightly) 0.26.0.dev20260130, on Meridian's default JAX
backend. A cold run takes about 30 minutes; saved models are reused when the configuration has not
changed.

## The data

Meridian's simulated [geo dataset](https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv):
40 geos, 156 weeks (2021-01-25 to 2024-01-15), five paid channels, two controls, a promotion and one
organic channel. It is committed in `data/raw/` (from a fixed version of Meridian's repository) and
checked before use. What we learned from it: [reports/eda.md](reports/eda.md).

- **Data quality:** the data are complete and clean.
- **Trend and seasonality:** revenue grows about 1.5% a year, with no meaningful yearly seasonality.
- **Noise:** weekly revenue is noisy, about 30% up or down per geo and 6.9% nationally, which limits
  how precise any model can be.
- **Media signal:** media's effect is hard to see in the raw data, so the model has to pool geos to
  find it.
- **Simulated conversions:** conversions are inflated in this dataset (about 19 per person per week),
  so revenue per conversion (about 0.02) looks low. Revenue itself is on a realistic scale.

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
data/                     the raw data file and the lift-test format
scripts/common.py         shared code: paths and data loading, convergence check
scripts/model_spec.py     the model definition (from config/model.toml)
scripts/01_eda/           data checks and exploratory tests
scripts/02_fit/           fit the model, check that it converged
scripts/03_validate/      forecast tests on shorter histories, Meridian health check, acceptance gates
scripts/04_scenarios/     budget scenarios, the hypothetical lift test and calibration
scripts/05_report/        channel results and the results report
docker/                   container checks: the EDA reproduces, sampling runs
models/                   fitted models (not committed) and their settings
outputs/                  the numbers behind every report, in the folder of the phase that wrote them
reports/                  eda.md, results.md, figures, Meridian reports
Makefile, Dockerfile, requirements.lock
```
