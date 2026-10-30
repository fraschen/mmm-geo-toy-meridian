"""Draws the charts in reports/results.md from outputs/ (generated upstream).

The report text is hand-written; this script only (re)draws its figures, into
reports/figures/diagnostics/ and reports/figures/measurement/. Figures use matplotlib
with the project palette; Meridian's own HTML reports are in reports/meridian/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (CHANNEL_COLOR, CHANNELS, FIGURES, INK_2, MUTED, OUTPUTS, PALETTE,
                    style_matplotlib)

style_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

FD, FM = FIGURES / "diagnostics", FIGURES / "measurement"
FD.mkdir(parents=True, exist_ok=True)
FM.mkdir(parents=True, exist_ok=True)


def J(path):
  """Load a JSON file from outputs/."""
  with open(OUTPUTS / path) as f:
    return json.load(f)


def save(fig, folder, name):
  fig.savefig(folder / name)
  plt.close(fig)
  return f"figures/{folder.name}/{name}"


def dates(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def load_all():
  d = {k: J(p) for k, p in {
      "cv": "03_validate/cv_metrics.json", "contrib": "05_report/contributions.json",
      "cal": "04_scenarios/calibration.json",
      "main": "04_scenarios/scenario_main.json", "stress": "04_scenarios/scenario_stress.json"}.items()}
  d["conv"] = J("02_fit/convergence.json")
  d["rhat"] = pd.DataFrame([{"variable": v, **p} for v, p in d["conv"]["per_variable"].items()])
  d["nat"] = pd.read_csv(OUTPUTS / "05_report" / "national_weekly.csv", parse_dates=["time"])
  d["wc"] = pd.read_csv(OUTPUTS / "05_report" / "weekly_contribution.csv", parse_dates=["time"])
  d["rpp"] = pd.read_csv(OUTPUTS / "05_report" / "roi_prior_posterior.csv")
  d["hill"] = pd.read_csv(OUTPUTS / "05_report" / "hill_curves.csv")
  d["resp"] = pd.read_csv(OUTPUTS / "05_report" / "response_curves.csv")
  d["res"] = pd.read_csv(OUTPUTS / "05_report" / "channel_results.csv")
  d["cvf"] = pd.read_csv(OUTPUTS / "03_validate" / "cv_national_forecasts.csv", parse_dates=["time"])
  d["cvroi"] = pd.read_csv(OUTPUTS / "03_validate" / "cv_roi.csv")
  d["rba"] = pd.read_csv(OUTPUTS / "04_scenarios" / "roi_before_after.csv")
  return d


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
                    label="94% interval of the model's mean")
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
                                     ("after", PALETTE[0], "After (lift test)"))):
    xx = np.arange(5) + (j - .5) * .3
    ax.errorbar(xx, r[f"{k}_mean"], yerr=[r[f"{k}_mean"] - r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] - r[f"{k}_mean"]],
                fmt="o", ms=5, color=col, lw=1.5, label=lab)
  t = d["cal"]["test"]
  i = CHANNELS.index(t["channel"])
  ax.errorbar([i + .3], [t["point_estimate_roi"]], yerr=[[1.88 * t["standard_error"]], [1.88 * t["standard_error"]]],
              fmt="s", ms=5, color=PALETTE[1], lw=1.5, label="Lift test (+/-1.88 SE)")
  ax.axhline(1, color=MUTED, ls="--", lw=1)
  ax.set_xticks(range(5), CHANNELS)
  ax.set_ylabel("ROI (mean, 94% CI)")
  ax.legend(loc="upper left", fontsize=7)
  ax.set_title("Calibration with a lift test: ROI before vs after")
  return save(fig, FM, "calibration.png")


def main():
  d = load_all()
  # Diagnostics
  fig_convergence(d)
  fig_fit(d, FD, "insample_fit.png", False)
  fig_residuals(d)
  fig_cv(d)
  fig_roi_by_fold(d)
  fig_roi_prior_post(d, FD, "roi_prior_posterior.png")
  # Measurement
  fig_roi_mroi(d)
  fig_fit(d, FM, "fit_baseline.png", True)
  fig_waterfall(d)
  fig_share(d)
  fig_weekly(d)
  fig_saturation(d)
  fig_response(d)
  fig_budget(d)
  fig_cal(d)
  print(f"Wrote figures to {FD} and {FM}")


if __name__ == "__main__":
  main()
