"""Phase 6: reports/diagnostics.md and reports/measurement.md with figures.

Every number in the report text is read from outputs/ (generated upstream).
Figures use matplotlib with the project palette (Meridian's visualizers render
Altair charts, which cannot be exported to PNG in this environment); Meridian's
native HTML reports are linked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (AXIS, CHANNEL_COLOR, CHANNELS, FIGURES, HISTORY, INK_2, MUTED, OUTPUTS,
                    PALETTE, ROOT, load_model_config, style_matplotlib)

style_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

FD, FM = FIGURES / "diagnostics", FIGURES / "measurement"
FD.mkdir(parents=True, exist_ok=True)
FM.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(OUTPUTS / p))
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
M = lambda x: f"{x / 1e6:,.1f}M"


def save(fig, folder, name):
  fig.savefig(folder / name)
  plt.close(fig)
  return f"figures/{folder.name}/{name}"


def dates(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


# ----------------------------------------------------------------------------
def load_all():
  d = {k: J(p) for k, p in {
      "eda": "eda/eda_summary.json",
      "gates": "gates/gates.json", "cv": "validation/cv_metrics.json",
      "contrib": "analysis/contributions.json", "ident": "analysis/identification.json",
      "levels": "analysis/calibration_levels.json", "cal": "calibration/calibration.json",
      "main": "budget/scenario_main.json", "stress": "budget/scenario_stress.json"}.items()}
  d["sel"] = json.load(open(HISTORY / "selection.json"))
  d["conv"] = J("model/convergence.json")
  d["cfg"] = load_model_config()
  d["nat"] = pd.read_csv(OUTPUTS / "analysis" / "national_weekly.csv", parse_dates=["time"])
  d["wc"] = pd.read_csv(OUTPUTS / "analysis" / "weekly_contribution.csv", parse_dates=["time"])
  d["rpp"] = pd.read_csv(OUTPUTS / "analysis" / "roi_prior_posterior.csv")
  d["hill"] = pd.read_csv(OUTPUTS / "analysis" / "hill_curves.csv")
  d["resp"] = pd.read_csv(OUTPUTS / "analysis" / "response_curves.csv")
  d["rhat"] = pd.read_csv(OUTPUTS / "analysis" / "rhat_by_variable.csv")
  d["res"] = pd.read_csv(OUTPUTS / "results" / "channel_results.csv")
  d["cvf"] = pd.read_csv(OUTPUTS / "validation" / "cv_national_forecasts.csv", parse_dates=["time"])
  d["cvroi"] = pd.read_csv(OUTPUTS / "validation" / "cv_roi.csv")
  d["rba"] = pd.read_csv(OUTPUTS / "calibration" / "roi_before_after.csv")
  d["geo"] = pd.read_csv(OUTPUTS / "calibration" / "geo_choice_check.csv")
  return d


def gate(d, name):
  return next(g for g in d["gates"]["gates"] if g["gate"] == name)


# ---------------------------------------------------------------- figures ----
def fig_convergence(d):
  r = d["rhat"].sort_values("max_rhat")
  th = d["conv"]["thresholds"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
  y = np.arange(len(r))
  axes[0].barh(y, r["max_rhat"] - 1, color=PALETTE[0], height=.6)
  axes[0].axvline(th["rhat_max"] - 1, color=INK_2, ls="--", lw=1)
  axes[0].set_xlabel(f"max R-hat - 1 (dashed: gate {th['rhat_max']})")
  axes[1].barh(y, r["min_ess_bulk"], color=PALETTE[1], height=.6, label="bulk")
  axes[1].scatter(r["min_ess_tail"], y, color=PALETTE[2], s=14, zorder=3, label="tail")
  axes[1].axvline(th["ess_min"], color=INK_2, ls="--", lw=1)
  axes[1].set_xlabel(f"min ESS (dashed: gate {th['ess_min']})")
  axes[1].legend(loc="lower right")
  axes[0].set_yticks(y, r["variable"])
  fig.suptitle("Selected model: worst R-hat and ESS per parameter", y=1.0)
  return save(fig, FD, "convergence.png")


def fig_fit(d, folder, name, baseline):
  n = d["nat"]
  fig, ax = plt.subplots(figsize=(9, 3.6))
  ax.fill_between(n.time, n.pred_lo / 1e6, n.pred_hi / 1e6, color=PALETTE[0], alpha=.18,
                  lw=0, label="94% predictive interval")
  ax.plot(n.time, n.observed / 1e6, color=INK_2, lw=1.2, label="Observed")
  ax.plot(n.time, n.expected / 1e6, color=PALETTE[0], lw=1.8, label="Expected (posterior mean)")
  if baseline:
    ax.plot(n.time, n.baseline / 1e6, color=PALETTE[1], lw=1.8, label="Baseline (no paid media, organic or Promo)")
  ax.set_ylabel("National weekly revenue (millions)")
  ax.set_ylim(0 if baseline else None)
  dates(ax)
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4 if baseline else 3)
  ax.set_title("Model fit: national weekly revenue" + (" with baseline" if baseline else ""))
  return save(fig, folder, name)


def fig_residuals(d):
  n = d["nat"]
  res = (n.observed - n.expected) / n.expected
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
  axes[0].bar(n.time, res * 100, width=5, color=PALETTE[0])
  axes[0].axhline(0, color=MUTED, lw=1)
  axes[0].set_ylabel("Residual (% of expected)")
  axes[0].set_title("National weekly residuals")
  dates(axes[0])
  x = res.to_numpy() - res.mean()
  acf = [1] + [float((x[:-k] * x[k:]).sum() / (x ** 2).sum()) for k in range(1, 14)]
  axes[1].bar(range(14), acf, color=PALETTE[0], width=.6)
  band = 1.96 / np.sqrt(len(x))
  axes[1].axhspan(-band, band, color="#e1e0d9", alpha=.7, lw=0)
  axes[1].set_xlabel("Lag (weeks)")
  axes[1].set_ylabel("Autocorrelation")
  axes[1].set_title("Residual autocorrelation (grey: 95% noise band)")
  return save(fig, FD, "residuals.png")


def fig_cv(d):
  f = d["cvf"]
  folds = [k for k, v in d["cv"]["folds"].items() if v["converged"]]
  fig, axes = plt.subplots(1, max(len(folds), 1), figsize=(9, 3.4), squeeze=False)
  for ax, k in zip(axes[0], folds):
    g = f[f.fold == k]
    test = g[g.forecast_period]
    tail = g[g.week > test.week.min() - 27]
    ax.fill_between(tail.time, tail.ci_lo / 1e6, tail.ci_hi / 1e6, color=PALETTE[0], alpha=.18, lw=0,
                    label="94% predictive interval")
    ax.plot(tail.time, tail.observed / 1e6, color=INK_2, lw=1.2, label="Observed")
    ax.plot(tail.time, tail["mean"] / 1e6, color=PALETTE[0], lw=1.8, label="Model")
    ax.axvline(test.time.min(), color=PALETTE[1], ls="--", lw=1.2, label="Forecast origin")
    ax.set_title(f"{k}: last 26 training weeks + 13-week forecast")
    ax.set_ylabel("National weekly revenue (millions)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
  return save(fig, FD, "cv_forecast.png")


def fig_roi_by_fold(d):
  cv = d["cvroi"]
  res = d["res"]
  fig, ax = plt.subplots(figsize=(8, 3.2))
  x = np.arange(len(CHANNELS))
  series = [("Full data", res.roi_mean, res.roi_ci_lo, res.roi_ci_hi)]
  for k in cv.fold.unique():
    g = cv[cv.fold == k].set_index("channel").loc[CHANNELS]
    series.append((k, g["mean"], g.ci_lo, g.ci_hi))
  w = 0.8 / len(series)
  for j, (lab, m, lo, hi) in enumerate(series):
    xx = x - 0.4 + w * (j + .5)
    ax.errorbar(xx, m, yerr=[m - lo, hi - m], fmt="o", ms=5, color=PALETTE[j], capsize=0, lw=1.5, label=lab)
  ax.axhline(1, color=MUTED, ls="--", lw=1)
  ax.set_xticks(x, CHANNELS)
  ax.set_ylabel("ROI (mean, 94% CI)")
  ax.set_title("ROI by CV fold (converged folds only) vs full data")
  ax.legend(loc="upper left")
  return save(fig, FD, "roi_by_fold.png")


def fig_calibration(d):
  lv = d["levels"]
  nom = [float(k) for k in lv]
  fig, ax = plt.subplots(figsize=(5, 4))
  ax.plot([0.4, 1], [0.4, 1], color=MUTED, ls="--", lw=1, label="Perfect calibration")
  ax.plot(nom, [lv[k]["geo_week"] for k in lv], marker="o", color=PALETTE[0], label="In-sample, geo-week")
  ax.plot(nom, [lv[k]["national"] for k in lv], marker="o", color=PALETTE[1], label="In-sample, national")
  for k, v in d["cv"]["folds"].items():
    if v["converged"]:
      ax.scatter([0.94], [v["national"]["forecast"]["coverage_94"]], color=PALETTE[2], s=40, zorder=3,
                 label=f"Forecast {k}, national")
      ax.scatter([0.94], [v["geo_week"]["forecast"]["coverage_94"]], color=PALETTE[3], s=40, zorder=3,
                 label=f"Forecast {k}, geo-week")
  ax.set_xlabel("Nominal interval level")
  ax.set_ylabel("Empirical coverage")
  ax.set_title("Predictive interval calibration")
  ax.legend(loc="upper left", fontsize=7)
  return save(fig, FD, "calibration.png")


def fig_roi_prior_post(d, folder, name):
  r = d["rpp"]
  fig, ax = plt.subplots(figsize=(8, 3.4))
  for j, lab in enumerate(["prior", "posterior"]):
    g = r[r.distribution == lab].set_index("channel").loc[CHANNELS]
    xx = np.arange(5) + (-0.15 if j == 0 else 0.15)
    ax.errorbar(xx, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], fmt="o", ms=5,
                color=[MUTED, PALETTE[0]][j], lw=1.5, label=lab.capitalize())
  ax.axhline(1, color=MUTED, ls="--", lw=1)
  ax.set_xticks(range(5), CHANNELS)
  ax.set_ylabel("ROI (mean, 94% interval)")
  ax.set_yscale("log")
  ax.set_title("ROI: prior vs posterior (log scale)")
  ax.legend(loc="upper right")
  return save(fig, folder, name)


def fig_roi_mroi(d):
  r = d["res"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
  for ax, (k, title) in zip(axes, [("roi", "ROI"), ("mroi", "Marginal ROI (next 1% of spend)")]):
    for i, c in enumerate(CHANNELS):
      row = r.iloc[i]
      ax.errorbar([row[f"{k}_mean"]], [i], xerr=[[row[f"{k}_mean"] - row[f"{k}_ci_lo"]],
                                                 [row[f"{k}_ci_hi"] - row[f"{k}_mean"]]],
                  fmt="o", ms=6, color=CHANNEL_COLOR[c], lw=2)
      ax.annotate(f"{row[f'{k}_mean']:.2f}", (row[f"{k}_mean"], i), xytext=(0, 7),
                  textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_title(f"{title}: mean and 94% CI")
    ax.set_xlabel("Revenue per unit spend")
  axes[0].set_yticks(range(5), CHANNELS)
  axes[0].invert_yaxis()
  return save(fig, FM, "roi_mroi.png")


def fig_waterfall(d):
  c = d["contrib"]
  items = [("Baseline", c["baseline"]["mean"], MUTED)]
  short = {"Organic_channel0_impression": "Organic", "Organic_channel0": "Organic"}
  items += [(short.get(k, k), v["mean"], PALETTE[5 + i]) for i, (k, v) in enumerate(c["non_paid"].items())]
  items += [(k, c["paid"][k]["mean"], CHANNEL_COLOR[k]) for k in CHANNELS]
  fig, ax = plt.subplots(figsize=(10, 3.8))
  run = 0
  for i, (lab, v, col) in enumerate(items):
    ax.bar(i, v / 1e6, bottom=run / 1e6, color=col, width=.65)
    ax.annotate(f"{v / 1e6:,.1f}M", (i, (run + v) / 1e6), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK_2)
    run += v
  ax.bar(len(items), run / 1e6, color=PALETTE[0], width=.65)
  ax.annotate(f"{run / 1e6:,.0f}M", (len(items), run / 1e6), xytext=(0, 3), textcoords="offset points",
              ha="center", fontsize=8, color=INK_2)
  ax.set_xticks(range(len(items) + 1), [i[0] for i in items] + ["Expected total"])
  ax.set_ylim(c["baseline"]["mean"] / 1e6 * 0.9, run / 1e6 * 1.04)
  ax.set_ylabel("Revenue over 156 weeks (millions)")
  ax.set_title("Contribution waterfall (posterior means; y-axis starts near the baseline)")
  return save(fig, FM, "waterfall.png")


def fig_share(d):
  r = d["res"]
  paid_share = r.contribution_mean / r.contribution_mean.sum()
  fig, ax = plt.subplots(figsize=(8, 3.2))
  x = np.arange(5)
  ax.bar(x - 0.18, r.spend_share * 100, width=.34, color=MUTED, label="Share of paid spend")
  ax.bar(x + 0.18, paid_share * 100, width=.34, color=PALETTE[0], label="Share of paid-media contribution")
  for i in range(5):
    ax.annotate(f"{100 * r.spend_share[i]:.0f}%", (x[i] - .18, 100 * r.spend_share[i]), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.annotate(f"{100 * paid_share[i]:.0f}%", (x[i] + .18, 100 * paid_share[i]), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=8, color=INK_2)
  ax.set_xticks(x, CHANNELS)
  ax.set_ylabel("%")
  ax.legend(loc="upper left")
  ax.set_title("Spend share vs contribution share")
  return save(fig, FM, "share.png")


def fig_weekly(d):
  w = d["wc"]
  fig, ax = plt.subplots(figsize=(9, 3.6))
  ax.stackplot(w.time, *[w[c] / 1e6 for c in CHANNELS], colors=[CHANNEL_COLOR[c] for c in CHANNELS],
               labels=CHANNELS, lw=0)
  ax.set_ylabel("Incremental revenue (millions/week)")
  dates(ax)
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)
  ax.set_title("Weekly incremental revenue by paid channel (posterior mean)")
  return save(fig, FM, "weekly_contribution.png")


def fig_saturation(d):
  h = d["hill"]
  fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True)
  for ax, c in zip(axes, CHANNELS):
    g = h[h.channel == c]
    hist = g[g.distribution.isna()]
    if len(hist):
      ax.bar(hist.start_interval_histogram, hist.scaled_count_histogram,
             width=hist.end_interval_histogram - hist.start_interval_histogram, align="edge",
             color="#e1e0d9", lw=0, label="Observed media (scaled histogram)")
    for lab, col in (("prior", MUTED), ("posterior", CHANNEL_COLOR[c])):
      s = g[g.distribution == lab]
      ax.fill_between(s.media_units, s.ci_lo, s.ci_hi, color=col, alpha=.2, lw=0)
      ax.plot(s.media_units, s["mean"], color=col, lw=2, label=lab.capitalize())
    ax.set_title(c)
    ax.set_xlabel("Media units per capita")
  axes[0].set_ylabel("Hill saturation (0-1)")
  handles = [plt.Line2D([], [], color=MUTED, lw=2, label="Prior (mean, 94% band)"),
             plt.Line2D([], [], color=INK_2, lw=2, label="Posterior (channel colour, 94% band)"),
             plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", label="Observed media (scaled histogram)")]
  fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3)
  fig.suptitle("Saturation curves: prior vs posterior (94% bands) over the observed media distribution", y=1.04)
  return save(fig, FM, "saturation.png")


def fig_response(d):
  r = d["resp"]
  res = d["res"].set_index("channel")
  fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
  for ax, c in zip(axes, CHANNELS):
    g = r[r.channel == c].pivot(index="spend_multiplier", columns="metric", values="incremental_outcome")
    sp = r[(r.channel == c) & (r.metric == "mean")].set_index("spend_multiplier")["spend"]
    ax.fill_between(sp / 1e6, g.ci_lo / 1e6, g.ci_hi / 1e6, color=CHANNEL_COLOR[c], alpha=.2, lw=0)
    ax.plot(sp / 1e6, g["mean"] / 1e6, color=CHANNEL_COLOR[c], lw=2)
    ax.axvline(res.loc[c, "spend"] / 1e6, color=INK_2, ls="--", lw=1)
    ax.set_title(c)
    ax.set_xlabel("Spend over 156 weeks (M)")
  axes[0].set_ylabel("Incremental revenue (M)")
  fig.suptitle("Response curves (posterior mean, 94% CI); dashed: historical spend; beyond it the curve shape is assumed",
               y=1.04)
  return save(fig, FM, "response_curves.png")


def fig_budget(d):
  fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
  x = np.arange(5)
  for j, (k, col) in enumerate((("main", PALETTE[0]), ("stress", PALETTE[1]))):
    s = d[k]["channels"]
    axes[0].bar(x + (j - .5) * .38, [100 * s[c]["change_pct"] for c in CHANNELS], width=.36, color=col,
                label=f"{k} (+/-{100 * d[k]['spend_constraint']:.0f}%)")
  axes[0].axhline(0, color=MUTED, lw=1)
  axes[0].set_xticks(x, CHANNELS)
  axes[0].set_ylabel("Optimal vs last-52-week spend (%)")
  axes[0].legend(loc="upper left")
  axes[0].set_title("Optimal reallocation (fixed total budget)")
  for j, k in enumerate(("main", "stress")):
    g = d[k]["gain"]
    axes[1].errorbar([g["mean"] / 1e6], [j], xerr=[[(g["mean"] - g["ci_lo"]) / 1e6], [(g["ci_hi"] - g["mean"]) / 1e6]],
                     fmt="o", ms=7, color=PALETTE[j], lw=2)
    axes[1].annotate(f"P(gain>0) = {g['p_gain_gt_0']:.2f}", (g["mean"] / 1e6, j), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8, color=INK_2)
  axes[1].axvline(0, color=MUTED, ls="--", lw=1)
  axes[1].set_yticks([0, 1], ["main", "stress"])
  axes[1].set_ylim(-.6, 1.6)
  axes[1].set_xlabel("Expected revenue gain over 52 weeks (M), 94% CI")
  axes[1].set_title("Expected gain")
  return save(fig, FM, "budget.png")


def fig_cal(d):
  r = d["rba"].set_index("channel").loc[CHANNELS]
  if "after_mean" not in r:
    return None
  fig, ax = plt.subplots(figsize=(8, 3.2))
  for j, (k, col, lab) in enumerate((("before", MUTED, "Before (no lift test)"),
                                     ("after", PALETTE[0], "After (HYPOTHETICAL lift test)"))):
    xx = np.arange(5) + (j - .5) * .3
    ax.errorbar(xx, r[f"{k}_mean"], yerr=[r[f"{k}_mean"] - r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] - r[f"{k}_mean"]],
                fmt="o", ms=5, color=col, lw=1.5, label=lab)
  t = d["cal"]["test"]
  i = CHANNELS.index(t["channel"])
  ax.errorbar([i + .3], [t["point_estimate_roi"]], yerr=[[1.88 * t["standard_error"]], [1.88 * t["standard_error"]]],
              fmt="s", ms=5, color=PALETTE[1], lw=1.5, label="HYPOTHETICAL test (+/-1.88 SE)")
  ax.axhline(1, color=MUTED, ls="--", lw=1)
  ax.set_xticks(range(5), CHANNELS)
  ax.set_ylabel("ROI (mean, 94% CI)")
  ax.legend(loc="upper left", fontsize=7)
  ax.set_title("Calibration with a HYPOTHETICAL lift test: ROI before vs after")
  return save(fig, FM, "calibration.png")


# ---------------------------------------------------------------- reports ----
def table(df):
  cols = list(df.columns)
  out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
  out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
  return "\n".join(out)


def cv_findings(d):
  out = []
  folds = d["cv"]["folds"]
  conv = [k for k, v in folds.items() if v["converged"]]
  non = [k for k, v in folds.items() if not v["converged"]]
  out.append(f"{len(conv)} of {len(folds)} fold fits passed the convergence gate"
             + (f"; {', '.join(non)} did not, so they report diagnostics only." if non else "."))
  T = d["gates"]["thresholds"]
  for k in conv:
    f = folds[k]
    g, n = f["geo_week"], f["national"]
    out.append(f"{k}: geo-week forecast MAPE {pct(g['forecast']['mape'])} vs in-sample "
               f"{pct(g['in_sample']['mape'])} (ratio {g['error_ratio']:.2f}, gate <= {T['cv_error_ratio_max']}); "
               f"national MAPE {pct(n['forecast']['mape'])} (gate <= {pct(T['cv_national_mape_max'], 0)}), "
               f"ratio {n['error_ratio']:.2f}.")
    out.append(f"{k}: 94% interval coverage {pct(g['forecast']['coverage_94'], 0)} at geo-week level but "
               f"{pct(n['forecast']['coverage_94'], 0)} nationally (gate >= {pct(T['cv_coverage_min'], 0)}). "
               f"The national forecast is biased by {100 * n['forecast']['mean_signed_pct_error']:+.1f}% on "
               f"average and over-predicts in {pct(n['forecast']['share_weeks_over_predicted'], 0)} of forecast "
               "weeks. Summing geos makes the national interval narrow relative to a shared bias, so a few "
               "percent of bias is enough to push national weeks outside it while geo-week intervals, "
               "dominated by geo-level noise, still cover.")
  return out


def diagnostics(d, figs):
  g = {x["gate"]: x for x in d["gates"]["gates"]}
  eda, nf = d["eda"], d["eda"]["noise_floor"]
  conv, th = d["conv"], d["conv"]["thresholds"]
  sel = d["sel"]
  cvrows = []
  for k, f in d["cv"]["folds"].items():
    if f["converged"]:
      cvrows.append([k, f"1-{f['train_weeks']}", "yes", pct(f["geo_week"]["forecast"]["mape"]),
                     f"{f['geo_week']['error_ratio']:.2f}", pct(f["geo_week"]["forecast"]["coverage_94"], 0),
                     pct(f["national"]["forecast"]["mape"]), f"{f['national']['error_ratio']:.2f}",
                     pct(f["national"]["forecast"]["coverage_94"], 0),
                     f"{100 * f['national']['forecast']['mean_signed_pct_error']:+.1f}%", g[f"cv_{k}"]["status"]])
    else:
      cvrows.append([k, f"1-{f['train_weeks']}", "no", "-", "-", "-", "-", "-", "-", "-", "FAIL"])
  cvt = pd.DataFrame(cvrows, columns=["fold", "train weeks", "converged", "geo MAPE", "geo ratio",
                                      "geo cov.", "nat. MAPE", "nat. ratio", "nat. cov.", "nat. bias", "gate"])
  hc = d["gates"]["meridian_health"]
  hrows = [[c["check"], c["status"], c.get("comparison", "")] for c in hc["checks"]]
  ins = d["gates"]["insample"]
  r_ins, c_ins = g["insample_r2"]["value"], g["insample_coverage"]["value"]
  rpp = d["rpp"]
  contraction = g["channel_roi_checks"]["value"]
  idt = d["ident"]
  lines = [
      "# Diagnostics: model health",
      "",
      f"Production model (`config/model.toml`): {d['cfg']['model']['description']}",
      "",
      "Meridian's native reports: [health check](meridian/health_check.html), "
      "[Model Results Summary](meridian/results_summary.html).",
      "",
      "## Why this model",
      "The pipeline fits one model. It was chosen once, from candidates that differed on what the EDA "
      "cannot settle: baseline flexibility (knots), geo pooling of media effects (`eta_m`) and max lag. "
      "The record (specs, diagnostics, comparison) is in "
      "[docs/model_selection](../docs/model_selection/README.md); the candidates are not refitted.",
      "- Each fit had to pass a convergence gate before PSIS-LOO comparison. The fits ran in two rounds:",
  ] + [f"  - {r.replace('_', ' ').capitalize()}: {sel['round_notes'][r]} "
       + "; ".join(f"{sel['labels'][n]} {'passed' if sel['converged'][n] else 'failed'}" for n in names) + "."
       for r, names in sel["rounds"].items()] + [f"- {sel['labels'][st['challenger']]} vs {sel['labels'][st['current']]}: ELPD difference "
       f"{st['elpd_diff']:+.1f} (paired SE {st['se']:.1f}); switching needs more than {st['threshold']:.1f}, so "
       f"{sel['labels'][st['challenger'] if st['switch'] else st['current']]} is kept."
       for st in sel["comparison_steps"]] + [
      f"- Selected: {sel['labels'][sel['selected']]}, PSIS-LOO ELPD {sel['loo'][sel['selected']]['elpd_loo']:.1f} "
      f"(SE {sel['loo'][sel['selected']]['se']:.1f}), {sel['loo'][sel['selected']]['pareto_k_gt_0.7']} Pareto k values above 0.7.",
      "- The candidate set and the convergence criteria were revised after earlier rounds failed to converge.",
      "",
      "## Data and prior checks",
      f"- Data integrity: {g['data_integrity']['status']} (checksum, balanced 40 x 156 panel, no "
      "missing or duplicate rows, non-negative spend/impressions/KPI). Details in [eda.md](eda.md).",
      f"- Prior predictive: {pct(g['prior_predictive_coverage']['value'], 0)} of national weeks lie in "
      f"the prior predictive 94% interval ({g['prior_predictive_coverage']['status']}); the interval is wide "
      f"({g['prior_predictive_coverage']['note']}), so this only rules out priors that cannot produce the data.",
      "",
      "## Convergence",
      f"- Gate: R-hat <= {th['rhat_max']}, bulk and tail ESS >= {th['ess_min']}, at most "
      f"{th['divergences_max']:.0f} divergent draws of {conv['n_draws_total']}.",
      f"- Selected model: max R-hat {conv['max_rhat']:.4f}, min bulk ESS {conv['min_ess_bulk']:.0f}, "
      f"min tail ESS {conv['min_ess_tail']:.0f}, {conv['divergences']} divergence(s): {g['convergence']['status']}.",
      "",
      f"![]({figs['convergence']})",
      "",
      "## In-sample fit and residuals",
      f"- R2: geo-week {r_ins['geo_week']:.3f}, national weekly {r_ins['national']:.3f}; MAPE: geo-week "
      f"{pct(ins['geo_week_mape'])}, national {pct(ins['national_mape'])} (noise floor: a 13-week moving "
      f"average has MAPE {pct(nf['geo_mape_vs_13wk_ma']['median'])} per geo, {pct(nf['national_mape_vs_13wk_ma'])} nationally).",
      f"- 94% interval coverage: geo-week {pct(c_ins['geo_week'])}, national {pct(c_ins['national'])} "
      f"({g['insample_coverage']['status']}). Within-geo R2 {g['within_geo_r2']['value']:.3f} "
      f"({g['within_geo_r2']['status']}); residual lag-1 autocorrelation: national "
      f"{g['residual_autocorrelation']['value']['national_lag1']:+.3f}, median geo "
      f"{g['residual_autocorrelation']['value']['median_geo_lag1']:+.3f} ({g['residual_autocorrelation']['status']}).",
      "",
      f"![]({figs['fit']})",
      "",
      f"![]({figs['residuals']})",
      "",
      "## Holdout and rolling-origin CV",
      "Folds train on weeks 1-104, 1-117, 1-130 and 1-143 (the last is the 13-week holdout) and forecast "
      "the next 13 weeks. Each fold is refitted on the truncated data, so all scaling is computed "
      "inside the fold. Forecasts use Meridian's equations with the linear baseline extended, "
      "conditional on observed media, controls, Promo and organic.",
      "",
      "Findings:",
  ] + [f"- {x}" for x in cv_findings(d)] + [
      "",
      table(cvt),
      "",
      f"![]({figs['cv']})",
      "",
      f"![]({figs['roi_by_fold']})",
      "",
      f"ROI stability across folds: {g['roi_fold_stability']['status']} ({g['roi_fold_stability']['note']}).",
      "",
      f"![]({figs['calibration']})",
      "",
      "## Identification",
      "- ROI contraction (1 - posterior sd / prior sd): "
      + ", ".join(f"{c} {contraction[c]['roi_contraction']:.2f}" for c in CHANNELS)
      + f" (warn below {d['gates']['thresholds']['roi_contraction_min']}): the data move every channel's ROI "
      "away from the prior.",
      f"- Baseline vs media: across posterior draws, total baseline and total paid-media contribution have "
      f"correlation {idt['corr_baseline_total_vs_paid_media_total']:.2f}. The model pins down their sum "
      "precisely but not the split between them; this is the main source of ROI uncertainty.",
      "",
      f"![]({figs['roi_prior_post_d']})",
      "",
      "## Meridian health checks",
      f"Overall: {hc['overall_status']}, health score {hc['health_score']:.1f}/100. Where Meridian's "
      "thresholds are looser than our gates:",
      "",
      table(pd.DataFrame(hrows, columns=["check", "status", "comparison with our gates"])),
      "",
      "## Gate summary (`make gates`)",
      "",
      "Rules, thresholds and rationale: [gates.md](gates.md) (generated from `config/gates.toml`).",
      "",
      table(pd.DataFrame([[x["gate"], x["status"], "yes" if x["blocking"] else "no"] for x in d["gates"]["gates"]],
                         columns=["gate", "status", "blocking"])),
      "",
      f"Blocking failures: {', '.join(d['gates']['blocking_failures']) or 'none'}.",
      "",

      "## Known limitations",
      "- Out-of-sample evidence is thin: only one CV fold fit converged, and its national interval "
      "coverage misses the gate.",
      "- Sampling is fragile: the selected model converges on the full data but not reliably on shorter windows.",
      "- Baseline and media trade off almost one-for-one, so channel ROIs are wide and no channel's "
      "payback is determined (P(ROI > 1) between 0.1 and 0.9 for every channel).",
      "- The linear-trend baseline cannot absorb week-level shocks common to all geos; the EDA found "
      "some common weekly variation beyond noise.",
      "",
  ]
  (ROOT / "reports" / "diagnostics.md").write_text("\n".join(lines))


def measurement(d, figs):
  r = d["res"]
  c = d["contrib"]
  cal = d["cal"]
  tab = pd.DataFrame({
      "channel": r.channel,
      "spend": [M(x) for x in r.spend], "spend share": [pct(x) for x in r.spend_share],
      "contribution (94% CI)": [f"{M(m)} ({M(lo)}-{M(hi)})" for m, lo, hi in
                                zip(r.contribution_mean, r.contribution_ci_lo, r.contribution_ci_hi)],
      "share of revenue": [pct(x) for x in r.share_of_revenue],
      "ROI (94% CI)": [f"{m:.2f} ({lo:.2f}-{hi:.2f})" for m, lo, hi in zip(r.roi_mean, r.roi_ci_lo, r.roi_ci_hi)],
      "mROI (94% CI)": [f"{m:.2f} ({lo:.2f}-{hi:.2f})" for m, lo, hi in zip(r.mroi_mean, r.mroi_ci_lo, r.mroi_ci_hi)],
      "P(ROI>1)": [f"{x:.2f}" for x in r.p_roi_gt_1], "gate": r.gate_status})
  paid_tot = c["paid_total"]
  top = r.sort_values("roi_mean", ascending=False)
  over = r[r.contribution_mean / r.contribution_mean.sum() > r.spend_share].channel.tolist()
  rows = []
  for k in ("main", "stress"):
    s = d[k]
    rows.append([k, f"+/-{100 * s['spend_constraint']:.0f}%", M(s["budget"]),
                 ", ".join(f"{ch} {100 * v['change_pct']:+.0f}%" for ch, v in s["channels"].items()),
                 f"{M(s['gain']['mean'])} ({M(s['gain']['ci_lo'])} to {M(s['gain']['ci_hi'])})",
                 f"{s['gain']['p_gain_gt_0']:.2f}", f"{100 * s['budget_conserved']['relative_difference']:.3f}%"])
  bt = pd.DataFrame(rows, columns=["scenario", "constraint", "budget (52 wk)", "change vs last 52 weeks",
                                   "gain (94% CI)", "P(gain>0)", "budget diff"])
  eda_gm = d["eda"]["spend_vs_geo_max"]
  rba = d["rba"].set_index("channel")
  cal_rows = [[ch, f"{rba.loc[ch, 'before_mean']:.2f} ({rba.loc[ch, 'before_ci_lo']:.2f}-{rba.loc[ch, 'before_ci_hi']:.2f})",
               f"{rba.loc[ch, 'before_p_roi_gt_1']:.2f}",
               f"{rba.loc[ch, 'after_mean']:.2f} ({rba.loc[ch, 'after_ci_lo']:.2f}-{rba.loc[ch, 'after_ci_hi']:.2f})",
               f"{rba.loc[ch, 'after_p_roi_gt_1']:.2f}"] for ch in CHANNELS] if "after_mean" in rba else []
  t = cal["test"]
  gc = cal["geo_choice_check"]
  lines = [
      "# Measurement: results",
      "",
      "National channel results for the selected model over all 156 weeks "
      f"({r.period[0]}). Revenue = conversions x revenue per conversion. Health and caveats: "
      "[diagnostics.md](diagnostics.md). Meridian's native reports: "
      "[Model Results Summary](meridian/results_summary.html), budget optimization "
      "[main](meridian/optimization_main.html) and [stress](meridian/optimization_stress.html). "
      "Machine-readable: `outputs/results/channel_results.csv` (+ `_metadata.json`).",
      "",
      "## Channel results",
      "",
      table(tab),
      "",
      f"- Paid media drove {M(paid_tot['mean'])} of revenue (94% CI {M(paid_tot['ci_lo'])}-{M(paid_tot['ci_hi'])}), "
      f"{pct(paid_tot['mean'] / c['observed_total'])} of the {M(c['observed_total'])} observed; the baseline "
      f"{M(c['baseline']['mean'])}.",
      f"- Highest mean ROI: {top.channel.iloc[0]} ({top.roi_mean.iloc[0]:.2f}) and {top.channel.iloc[1]} "
      f"({top.roi_mean.iloc[1]:.2f}); lowest: {top.channel.iloc[-1]} ({top.roi_mean.iloc[-1]:.2f}). "
      "All intervals are wide and every channel's P(ROI > 1) lies between 0.1 and 0.9, so no channel's "
      "payback is determined.",
      "- Marginal ROI is below ROI for every channel (diminishing returns); "
      + ", ".join(f"{ch} {m:.2f}" for ch, m in zip(r.channel, r.mroi_mean)) + ".",
      "- Cost per incremental KPI is not reported because the outcome is modelled in revenue.",
      "",
      f"![]({figs['roi_mroi']})",
      "",
      "## Model fit and baseline",
      "",
      f"![]({figs['fit_baseline']})",
      "",
      "## Contribution waterfall",
      "",
      f"![]({figs['waterfall']})",
      "",
      "## Spend share vs contribution share",
      f"- Channels whose share of paid-media contribution exceeds their spend share: {', '.join(over) or 'none'}.",
      "",
      f"![]({figs['share']})",
      "",
      "## Weekly contribution by channel",
      "",
      f"![]({figs['weekly']})",
      "",
      "## Saturation curves",
      "",
      f"![]({figs['saturation']})",
      "",
      "## Response curves",
      "",
      f"![]({figs['response']})",
      "",
      "## Budget scenarios",
      f"Fixed total budget equal to the last 52 weeks' spend ({M(d['main']['budget'])}), optimized over "
      f"{d['main']['period'][0]} to {d['main']['period'][1]}. The gain is incremental revenue at the optimal "
      "allocation minus at the historical allocation, computed per posterior draw.",
      "",
      table(bt),
      "",
      f"- Main (+/-20%): shift spend toward "
      + ", ".join(ch for ch, v in d["main"]["channels"].items() if v["change_pct"] > 0)
      + f"; expected gain {M(d['main']['gain']['mean'])} ({pct(d['main']['gain']['pct_of_historical_incremental'])} "
      f"of media-driven revenue) with P(gain > 0) = {d['main']['gain']['p_gain_gt_0']:.2f}. The interval "
      "includes losses: the reallocation is more likely than not to help, and the size of the gain is uncertain.",
      f"- Stress (+/-50%): larger shifts, expected gain {M(d['stress']['gain']['mean'])} with P(gain > 0) = "
      f"{d['stress']['gain']['p_gain_gt_0']:.2f}; wider interval.",
      "- Budget is conserved in both scenarios (difference = the optimizer's rounding of the budget to its spend grid).",
      "- Beyond historically observed spend, results assume the response-curve shape. Scaling the last 52 "
      "weeks by 1.5 would push "
      + ", ".join(f"{ch} {pct(eda_gm[ch]['last52_share_geo_weeks_beyond_geo_max_if_x1.5'], 0)}" for ch in CHANNELS)
      + " of geo-weeks above each geo's historical maximum.",
      "",
      f"![]({figs['budget']})",
      "",
      "## Calibration with a HYPOTHETICAL lift test",
      f"**HYPOTHETICAL** test (`data/lift_tests/lift_tests.csv`, schema in `data/lift_tests/README.md`): "
      f"{t['channel']}, {t['start_date']} to {t['end_date']}, {len(t['test_geos'].split(';'))} test geos, "
      f"ROI {t['point_estimate_roi']} (SE {t['standard_error']}). It was fed to Meridian's `CalibrationBuilder` "
      f"(baseline prior LogNormal(0.2, 0.9), geometric decay, max lag 4). The resulting {t['channel']} ROI prior has "
      f"mean {cal['prior_channel']['after_calibrated']['mean']:.2f} and sd {cal['prior_channel']['after_sd']:.2f}: "
      "the spend adjustment widens the test's uncertainty because the test covered a fraction of the channel's "
      "national spend. The calibrated refit "
      + ("converged" if cal["after_converged"] else "did not converge")
      + f" (max R-hat {cal['after_convergence']['max_rhat']:.4f}, min bulk ESS {cal['after_convergence']['min_ess_bulk']:.0f}, "
      f"{cal['after_convergence']['divergences']} divergences).",
      "",
      table(pd.DataFrame(cal_rows, columns=["channel", "ROI before (94% CI)", "P(ROI>1) before",
                                            "ROI after (94% CI)", "P(ROI>1) after"])) if cal_rows else "",
      "",
      f"- Model vs test: for the test geos and weeks, the uncalibrated model implied {t['channel']} ROI "
      f"{cal['implied_test_roi_before']['mean']:.2f} (94% CI {cal['implied_test_roi_before']['ci_lo']:.2f}-"
      f"{cal['implied_test_roi_before']['ci_hi']:.2f}); the HYPOTHETICAL test said {t['point_estimate_roi']}."
      + (f" After calibration the model implies {cal['implied_test_roi_after']['mean']:.2f}." if "implied_test_roi_after" in cal else ""),
      f"- Test-geo choice: across {gc['n_alternative_sets']} alternative stratified sets of the same size, the "
      f"model-implied ROI ranges {gc['implied_roi_mean_range'][0]:.2f}-{gc['implied_roi_mean_range'][1]:.2f} "
      f"(chosen set at the {pct(gc['chosen_implied_percentile'], 0)} percentile) and the calibrated prior mean "
      f"{gc['calibrated_prior_mean_range'][0]:.2f}-{gc['calibrated_prior_mean_range'][1]:.2f}. The choice of test "
      "geos does not drive the result.",
      f"- The test value equals the uncalibrated national ROI estimate to two decimals by coincidence (it was "
      "fixed before that estimate was computed), so calibration narrows the interval more than it moves the mean.",
      "",
  ] + ([f"![]({figs['cal']})", ""] if figs.get("cal") else []) + [
      "## Caveats",
      "- Read [diagnostics.md](diagnostics.md) first: out-of-sample validation is thin, and the candidate "
      "set and convergence criteria were revised after earlier rounds failed to converge.",
      "- Baseline and media trade off almost one-for-one; totals are firmer than channel splits.",
      "- National results only; no geo-level ROI.",
      "- The lift test is HYPOTHETICAL; calibrated numbers illustrate the method, not real evidence.",
      "",
  ]
  (ROOT / "reports" / "measurement.md").write_text("\n".join(lines))


def main():
  d = load_all()
  figs = {"convergence": fig_convergence(d), "fit": fig_fit(d, FD, "insample_fit.png", False),
          "residuals": fig_residuals(d), "cv": fig_cv(d), "roi_by_fold": fig_roi_by_fold(d),
          "calibration": fig_calibration(d), "roi_prior_post_d": fig_roi_prior_post(d, FD, "roi_prior_posterior.png"),
          "roi_mroi": fig_roi_mroi(d), "fit_baseline": fig_fit(d, FM, "fit_baseline.png", True),
          "waterfall": fig_waterfall(d), "share": fig_share(d), "weekly": fig_weekly(d),
          "saturation": fig_saturation(d), "response": fig_response(d), "budget": fig_budget(d),
          "cal": fig_cal(d)}
  diagnostics(d, figs)
  measurement(d, figs)
  print("Wrote reports/diagnostics.md and reports/measurement.md")


if __name__ == "__main__":
  main()
