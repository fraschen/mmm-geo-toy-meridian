You are building a geo-level Bayesian marketing mix model (MMM) with Google Meridian, delivered as a reproducible
GitHub repo. Use Meridian's official agent skills throughout (already installed in `.claude/skills/`). Stop at the
checkpoints marked ⏸ for my approval.

## Skills
- `meridian-doc-consultant`: for any conceptual question (priors, knots, ROI calibration, health checks). Answer
  from the official docs, not from memory.
- `meridian-model-building`: data loading, column mapping, ModelSpec, EDA, fitting and saving
  (`meridian_model.binpb`).
- `meridian-result-visualization`: Model Results Summary and health checks for fitted models.
- `meridian-budget-optimization`: budget scenarios.
- `meridian-scenario-planner`: optional. It uploads to Google Sheets / Looker Studio; ask me before using it.
Follow each skill's interactive checkpoints; they are mandatory and add to the ⏸ checkpoints below. Keep the
Python scripts the skills generate, put them in the repo, and call them from the Makefile so `make all`
reproduces them. Where this prompt asks for something a skill does not cover (EDA tests, candidate selection,
CV, gates, calibration, markdown reports), write it yourself, using the same Meridian version and the skills'
loading and serialization conventions.

## Data
Meridian's simulated geo dataset:
https://github.com/google/meridian/blob/main/meridian/data/simulated_data/csv/geo_all_channels.csv
40 geos × 156 weeks; 5 paid channels (ChannelN_impression, ChannelN_spend), controls (competitor_sales_control,
sentiment_score_control), Promo, Organic_channel0_impression, conversions (KPI), revenue_per_conversion,
population. Download it with a checksum check; never modify the raw file.

## Deliverables
1. Code that reproduces everything from one command (`make all`) and from Docker (`make docker`).
2. A channel-level results file (CSV + metadata JSON with column definitions and package versions). Per channel:
   contribution, ROI and marginal ROI with 94% credible intervals, P(ROI > 1), spend, spend share, share of
   revenue, ROI range across CV folds, gate status, period and units.
3. `reports/diagnostics.md` (model health) and `reports/measurement.md` (results), with figures, plus the
   skills' native HTML reports (results summary, optimization) linked from them.
4. A README containing: verdict, reproduce, data, EDA, candidates and selection result, Results (a short overview
   of diagnostics, measurement insights and budget recommendations, each linking to the detailed report), gates,
   what the data can't support, next steps, repository layout.

## Environment
- The environment is `.venv` (Python 3.11) with `google-meridian` already installed; the skills use that
  interpreter. Pin every dependency in a lockfile.
- Use Meridian's JAX backend. In `google-meridian` 2.x JAX is the default and the TensorFlow backend is
  deprecated: do not set `MERIDIAN_BACKEND=tensorflow`. Meridian's API changes between versions: as the skills
  instruct, read the installed package's source and signatures before using any class or argument. Record the
  Meridian, JAX, jaxlib and TensorFlow Probability versions and the backend in the README.
- TensorFlow is installed only as a transitive dependency; do not use it in project code.

## Phase 1: EDA (meridian-model-building's EDA step, extended)
Run the skill's EDA. Then add `make eda`, writing a summary JSON and figures covering:
- data sufficiency, panel balance, missing and negative values;
- whether spend = impressions × a constant CPM per channel;
- national trend with a HAC-robust test;
- yearly seasonality tested against chance (simulate how much variance week-of-year means explain in pure noise);
- noise floor: within-geo weekly CV and lag-1 autocorrelation;
- within-geo spend–revenue correlation at lags 0–8 (to inform max lag);
- within-geo channel collinearity and VIF;
- revenue and spend vs population;
- where geo-week spend sits relative to each geo's max, and the share of zero weeks;
- controls' correlation with revenue and with spend (confounders?);
- heterogeneity of control effects across geos (per-geo regressions + Cochran's Q / I²).
Every EDA number in the README must be reproduced by `make eda`.

## Phase 2: candidate specifications ⏸
Using meridian-model-building's ModelSpec step (and meridian-doc-consultant for prior choices), propose 2–3
specifications justified by the EDA alone. They should differ on what the EDA cannot settle (e.g. geo pooling of
media effects, how controls enter, knots, max lag). For each, state the EDA evidence and the settings: media
effects distribution, ROI vs coefficient priors, knots, max lag, controls, population handling.
Fix the selection protocol before fitting:
- a fit that fails any convergence check (R-hat ≤ 1.01, bulk and tail ESS ≥ 400, 0 divergences where reported)
  stops without results and is dropped; no reruns with more samples;
- among converged candidates, select by PSIS-LOO with pointwise values saved; prefer the simpler candidate unless
  the more complex one is better by more than 2 SE of the difference.
⏸ Show me the candidates and protocol, and wait. Do not fit anything before I approve.

## Phase 3: fit and select (meridian-model-building)
- Fit each candidate with a fixed seed, 4 chains and moderate draws, and save each as its own `.binpb`. Do not
  change sampler settings after seeing results without my approval.
- A non-converged fit records only its convergence diagnostics. Never report its ROI, fit or contributions
  anywhere.
- If only one candidate converges, say plainly that it won by convergence, not by fit.
⏸ Report convergence and selection, and wait.

## Phase 4: validation of the selected spec
- A 13-week holdout plus rolling-origin CV: train on weeks 1–104, 1–117, 1–130 and 1–143, each forecasting the next
  13 weeks. Compute any data-derived scaling inside each fold.
- Score at geo-week and national level: MAPE, forecast-vs-in-sample error ratio, 94% interval coverage. If
  national coverage fails while geo-week coverage passes, explain the gap from the data (e.g. bias per fold),
  not from a guess.

## Phase 5: health checks and automated gates
- Run meridian-result-visualization's health checks and Model Results Summary on the selected model.
- Add `make gates` (exits non-zero on any blocking failure), gating the selected spec only, with no
  cross-specification ROI agreement gate:
  - data integrity;
  - prior predictive coverage;
  - convergence: R-hat, ESS, divergences;
  - in-sample R² and interval coverage;
  - warnings: within-geo R², residual autocorrelation;
  - holdout and each fold: error ratio ≤ 1.25, national weekly MAPE ≤ 10%, coverage ≥ 80%;
  - ROI stability across folds: > 2× warns, > 5× fails;
  - per channel: ROI contraction vs prior (warn if prior-driven) and P(ROI > 1) ≥ 0.9 or ≤ 0.1 (warn if
    undetermined).
- Include Meridian's health check outcomes in the gate report, noting where its thresholds are looser than ours.
- Set accuracy thresholds against the EDA's noise floor, and document any change.

## Phase 6: reports
Build on the skills' outputs and fill the gaps.

`reports/diagnostics.md`:
- how the model was chosen;
- data and prior checks;
- convergence;
- in-sample fit and residuals;
- holdout/CV: a generated findings list, a per-fold table, ROI by fold, forecast plots, national calibration;
- identification: prior vs posterior ROI, the baseline–media correlation;
- Meridian health checks;
- known limitations.

`reports/measurement.md`:
- the channel results table, and ROI + marginal ROI;
- model fit with a baseline line;
- a contribution waterfall;
- spend share vs contribution share;
- weekly contribution by channel;
- saturation curves (prior vs posterior) with histograms of observed spend;
- response curves;
- budget scenarios;
- calibration;
- caveats.

Use Meridian's visualizers where they exist, and matplotlib otherwise, with one palette and no dual axes. Review
every figure for label collisions.
Rules:
- national channel results only (no geo-level ROI);
- cost per incremental KPI only if the modeled outcome is not already revenue;
- every number in report text generated from data.

## Phase 7: budget scenarios (meridian-budget-optimization)
Fixed total budget equal to the last 52 weeks' spend; a main scenario with per-channel spend constraints of ±20%,
and a stress scenario of ±50%. Report the optimal allocation, the expected revenue gain with its credible
interval and P(gain > 0), and verify that the budget is conserved. Present uncertainty as information for the
business, not as a reason to withhold a scenario. State once that beyond historically observed spend, results
assume the response-curve shape.

## Phase 8: calibration with a hypothetical lift test
Invent a lift test for the largest channel with undetermined payback, and label it HYPOTHETICAL everywhere.
Generate it with a script, in a documented file format that a real result could replace. Calibrate the way
Meridian intends (ROI priors informed by the test; confirm with meridian-doc-consultant), and refit with
meridian-model-building. Report ROI and P(ROI > 1) before vs after for every channel, and what the model implied
for the test vs what the test said. Check whether the choice of test geos drives the result.

## Phase 9: reproducibility and README
- `make all` runs every phase (including the skill-generated scripts) and is idempotent on a fresh clone; keep the
  machine awake during long runs on macOS.
- A Dockerfile from the lockfile: verify that the image builds, EDA reproduces, and sampling runs in the container.
- README verdict up front: is the model good enough for a client, and why, with the caveats that must accompany
  the numbers.
- README Results section: a very short overview of the diagnostics, the measurement insights and the budget
  recommendations, each linking to the detailed report.

## Working style
- Report outcomes faithfully. Recompute any number that looks inconsistent before reporting it, and correct
  explanations that turn out wrong.
- Commit locally at each milestone; ask me before creating or pushing a GitHub repo, and before anything that
  uploads data externally (e.g. the scenario planner).

## Optional comparison
If I give you the path to an earlier pymc-marketing study of the same data (`/Users/fs/code/mmm-geo-toy-pymc-marketing`),
add an appendix comparing channel ROI, contributions and diagnostics, and explain differences by modeling choice
rather than by framework.
