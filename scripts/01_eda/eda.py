"""Extended EDA (`make eda`): summary JSON and figures.

Data checks and exploratory tests that inform the model specification, plus
Meridian's built-in data checks. Writes outputs/01_eda/eda_summary.json and the
figures used in reports/eda.md (hand-written from these numbers).

Revenue is conversions x revenue_per_conversion (the modelled KPI is
conversions, with revenue_per_conversion turning it into revenue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from meridian.model.eda import meridian_eda
from meridian.model.eda.constants import NATIONALIZE
from meridian.model.model import Meridian
from scipy import stats

from common import (CHANNEL_COLOR, CHANNELS, CONTROLS, FIGURES, GEO, INK_2,
                    KPI, MUTED, NON_MEDIA_TREATMENTS, ORGANIC_MEDIA, OUTPUTS,
                    PALETTE, POPULATION, REVENUE_PER_KPI, ROOT, SEED, SPEND_COLS,
                    IMPRESSION_COLS, TIME, build_input_data, load_model_config,
                    load_raw, style_matplotlib, write_json)
from model_spec import make_spec

style_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG = FIGURES / "eda"
FIG.mkdir(parents=True, exist_ok=True)
OUT = OUTPUTS / "01_eda"
N_SIM = 10_000
MAX_LAG = 8
SEASON = 52

DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#eb6834", "#f3f2ee", "#2a78d6"])


def savefig(fig, name):
  fig.savefig(FIG / name)
  plt.close(fig)
  return f"reports/figures/eda/{name}"


def save_chart(chart, name):
  """Save one of Meridian's EDA charts (Altair) as a PNG."""
  chart.save(FIG / name, scale_factor=2)
  return f"reports/figures/eda/{name}"


def demean(df, cols, by):
  """Subtract each group's own average, e.g. each geo's mean ("within-geo" values)."""
  return df[cols] - df.groupby(by)[cols].transform("mean")


def severity(outcome):
  """Worst severity among a check's findings: INFO (fine) or REVIEW (flagged)."""
  levels = [f.severity for f in outcome.findings]
  return max(levels, key=lambda s: s.value).name if levels else "INFO"


def national_series(df):
  return df.groupby(TIME)["revenue"].sum()


def woy_r2(Y, period=SEASON):
  """Share of variance explained by week-of-year means.

  Y has shape (..., time): the last axis (-1) is time, so Y may be one series
  (weeks,) or many (series, weeks).
  """
  T = Y.shape[-1]
  g = np.arange(T) % period                  # week of the year of each week: 0..51, 0..51, ...
  Yc = Y - Y.mean(-1, keepdims=True)         # Y centered around the mean of each series
  # Average of each week of the year (e.g. all "week 10"s), for every series at once.
  means = np.stack([Yc[..., g == k].mean(-1) for k in range(period)], -1)
  fitted = means[..., g]                     # each week replaced by its week-of-year average
  # Share of the variation explained by the week-of-year averages (an R2 between 0 and 1).
  return (fitted ** 2).sum(-1) / (Yc ** 2).sum(-1)


def detrend(y):
  """Remove a straight-line trend: fit a line through the series and subtract it."""
  t = np.arange(len(y))
  b = np.polyfit(t, y, 1)                    # slope and intercept of the best straight line
  return y - np.polyval(b, t)


def meridian_eda_setup():
  """Meridian's EDA on the production model setup (priors only, no fitting).

  Also saves Meridian's own HTML report to reports/meridian/eda.html. Its charts sum
  geo-level series to national.
  """
  cfg = load_model_config()["sampler"]
  data = build_input_data()
  mmm = Meridian(input_data=data, model_spec=make_spec(data))
  mmm.sample_prior(n_draws=cfg["n_prior_draws"], seed=cfg["seed"])
  eda = meridian_eda.MeridianEDA(mmm)
  eda.generate_and_save_report(filename="eda.html", filepath=str(ROOT / "reports" / "meridian"))
  return eda


# --- The data ------------------------------------------------------------------
def sufficiency(df):
  geos = df[GEO].nunique()
  times = np.sort(df[TIME].unique())
  # Gaps between consecutive weeks, in days: a single value of 7 means no missing weeks.
  steps = np.unique(np.diff(times).astype("timedelta64[D]").astype(int))
  per_geo = df.groupby(GEO).size()         # number of rows per geo
  num = df.select_dtypes("number")
  spend_total = df[SPEND_COLS].sum()
  share = spend_total / spend_total.sum()
  last52 = df[df[TIME] > times[-53]]      # rows from the most recent 52 weeks
  nonzero_weeks = {c: int((df[s] > 0).sum()) for c, s in zip(CHANNELS, SPEND_COLS)}
  rpk = df[REVENUE_PER_KPI]
  return {
      "n_rows": len(df), "n_geos": geos, "n_weeks": len(times),
      "first_week": str(pd.Timestamp(times[0]).date()),
      "last_week": str(pd.Timestamp(times[-1]).date()),
      "time_step_days": steps.tolist(),
      "rows_per_geo_min": int(per_geo.min()), "rows_per_geo_max": int(per_geo.max()),
      # Every geo has the same number of rows, and rows = geos x weeks.
      "balanced_panel": bool(per_geo.nunique() == 1 and len(df) == geos * len(times)),
      "duplicate_geo_weeks": int(df.duplicated([GEO, TIME]).sum()),
      "missing_values_total": int(df.isna().sum().sum()),
      # Number of negative values, listed only for the columns that have any.
      "negative_values": {c: int((num[c] < 0).sum()) for c in num.columns
                          if (num[c] < 0).any()},
      "negative_values_note": "Controls are centred indices, so negatives are expected there.",
      "nonnegative_required_columns_ok": bool(
          (df[SPEND_COLS + IMPRESSION_COLS + ORGANIC_MEDIA + [KPI, POPULATION,
                                                             REVENUE_PER_KPI]] >= 0).all().all()),
      "population_constant_within_geo": bool((df.groupby(GEO)[POPULATION].nunique() == 1).all()),
      "spend_share": share.rename(lambda s: s.replace("_spend", "")).to_dict(),
      "total_spend": float(spend_total.sum()),
      "last_52_weeks_spend": float(last52[SPEND_COLS].sum().sum()),
      "last_52_weeks_spend_by_channel": last52[SPEND_COLS].sum().rename(
          lambda s: s.replace("_spend", "")).to_dict(),
      "total_revenue": float(df["revenue"].sum()),
      "nonzero_spend_geo_weeks": nonzero_weeks,
      "revenue_per_conversion": {
          "mean": float(rpk.mean()), "cv": float(rpk.std() / rpk.mean()),
          "min": float(rpk.min()), "max": float(rpk.max())},
  }


def summary_stats(df):
  """Mean, median, min, max and share of zero weeks for each input, per geo-week.

  Population is constant within a geo, so its statistics are across the geos.
  """
  columns = {"Revenue": "revenue"}
  columns.update({f"{c} spend": sp for c, sp in zip(CHANNELS, SPEND_COLS)})
  columns.update({c: c for c in CONTROLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA})
  out = {}
  for label, col in columns.items():
    x = df[col]
    out[label] = {"mean": float(x.mean()), "median": float(x.median()), "min": float(x.min()),
                  "max": float(x.max()), "share_zero": float((x == 0).mean())}
  pop = df.groupby(GEO)[POPULATION].first()
  out["Population (per geo)"] = {"mean": float(pop.mean()), "median": float(pop.median()),
                                 "min": float(pop.min()), "max": float(pop.max()), "share_zero": None}
  return out


def data_adequacy(eda):
  a = eda.eda_engine.check_data_param_ratio().analysis_artifacts[0]
  n_data = a.n_geos * a.n_times
  # Meridian's rough guide: data points per parameter, counting one parameter per geo
  # (minus the baseline geo), per knot, per control and per treatment.
  n_params = (a.n_geos - 1) + a.n_knots + a.n_controls + a.n_treatments
  return {"n_data_points": int(n_data), "n_parameters": int(n_params),
          "ratio": n_data / n_params, "n_knots": int(a.n_knots)}


# --- Revenue -------------------------------------------------------------------
def revenue_outliers(eda):
  """Unusual revenue weeks, and whether revenue varies at all."""
  std = eda.eda_engine.check_std()
  kpi = eda.eda_engine.check_overall_kpi_invariability()
  return {"n_outliers_flagged": next(int(len(a.outlier_df)) for a in std.analysis_artifacts
                                     if a.variable == "kpi_scaled"),
          "kpi_varies": severity(kpi) == "INFO",
          "kpi_scaled_sd": float(kpi.analysis_artifacts[0].kpi_stdev),
          "figure": save_chart(eda.plot_kpi_boxplot(geos=NATIONALIZE), "kpi_boxplot.png")}


def trend(nat):
  """National revenue per 52-week year and its year-on-year growth.

  The data start in late January, so each "year" is a block of 52 weeks, not a
  calendar year.
  """
  weeks = nat.index
  year = np.arange(len(nat)) // SEASON        # 0 for weeks 1-52, 1 for 53-104, 2 for 105-156
  revenue = nat.groupby(year).sum()           # one total per year
  growth = revenue.pct_change()               # vs the previous year (NaN for year 1)
  first, last = pd.Series(weeks).groupby(year).first(), pd.Series(weeks).groupby(year).last()
  table = pd.DataFrame({
      "first_week": first.dt.strftime("%Y-%m-%d"),
      "last_week": last.dt.strftime("%Y-%m-%d"),
      "revenue": revenue,
      "growth_vs_previous_year": growth,
  })
  table.index = table.index + 1                                 # years 1, 2, 3
  years = table.reset_index(names="year").to_dict(orient="records")

  # Weekly revenue with its 13-week average and a straight-line fit (the model's baseline shape).
  y = nat.to_numpy()
  t = np.arange(len(y))
  line = np.polyval(np.polyfit(t, y, 1), t)
  fig, ax = plt.subplots(figsize=(8, 3.2))
  ax.plot(weeks, y / 1e6, color=PALETTE[0], lw=1.2, label="National weekly revenue")
  ma = nat.rolling(13, center=True).mean()
  ax.plot(weeks, ma.to_numpy() / 1e6, color=PALETTE[1], lw=2, label="13-week average")
  ax.plot(weeks, line / 1e6, color=INK_2, lw=1.2, ls="--", label="Straight line")
  ax.set_ylabel("Revenue (millions)")
  ax.set_title("National weekly revenue")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
  return {"years": years, "figure": savefig(fig, "national_trend.png")}


def seasonality_test(df, nat):
  """Is there a week-of-year pattern beyond what random noise would produce?"""
  rng = np.random.default_rng(SEED)
  y = detrend(nat.to_numpy())              # national weekly revenue, linear trend removed
  T = len(y)                               # number of weeks (156)
  obs = float(woy_r2(y))                   # share of variation explained by week-of-year averages
  phi = float(np.corrcoef(y[:-1], y[1:])[0, 1])  # how much each week resembles the previous one

  # Benchmark 1: N_SIM series of independent random noise, each as long as the data.
  iid = woy_r2(rng.standard_normal((N_SIM, T)))

  # Benchmark 2: noise where each week carries over a share phi of the previous week (AR(1)).
  # e holds the fresh random shocks: one row per simulated series, one column per week.
  e = rng.standard_normal((N_SIM, T))
  ar = np.empty_like(e)
  # Week 1 has no previous week, so it starts from its shock, scaled to the series' long-run
  # spread (1 / sqrt(1 - phi^2)) so that it is no calmer than the weeks that follow.
  ar[:, 0] = e[:, 0] / np.sqrt(1 - phi ** 2)
  for t in range(1, T):
    ar[:, t] = phi * ar[:, t - 1] + e[:, t]   # carry-over from last week + new shock
  ar1 = woy_r2(ar)

  # Per geo: the same test on each geo's revenue, compared with the independent-noise benchmark.
  geo_y = df.pivot(index=GEO, columns=TIME, values="revenue").to_numpy()   # geos x weeks
  geo_r2 = woy_r2(np.apply_along_axis(detrend, 1, geo_y))                  # one value per geo
  # p-value per geo: share of noise series that score at least as high as that geo.
  # [:, None] and [None, :] line the 40 geos up against the N_SIM noise scores.
  geo_p = (iid[None, :] >= geo_r2[:, None]).mean(1)

  res = {
      "method": ("Detrended national revenue; R2 of week-of-year means "
                 f"(52 groups x 3 years) vs {N_SIM} simulated noise series of the "
                 "same length: iid Gaussian and AR(1) with the observed lag-1 autocorrelation."),
      "observed_r2": obs,
      "null_iid_mean": float(iid.mean()), "null_iid_p95": float(np.quantile(iid, .95)),
      "p_value_iid": float((iid >= obs).mean()),
      "detrended_lag1_autocorr": phi,
      "null_ar1_mean": float(ar1.mean()), "null_ar1_p95": float(np.quantile(ar1, .95)),
      "p_value_ar1": float((ar1 >= obs).mean()),
      "geo_r2_median": float(np.median(geo_r2)),
      "geos_significant_5pct_iid": int((geo_p < .05).sum()),
      "geos_expected_by_chance": 0.05 * len(geo_r2),
      "geos_p_value_binomial": float(stats.binom.sf(int((geo_p < .05).sum()) - 1, len(geo_r2), 0.05)),
  }
  res["seasonality_detected"] = bool(res["p_value_ar1"] < 0.05)

  fig, ax = plt.subplots(figsize=(7, 3.2))
  bins = np.linspace(0.1, 0.65, 56)
  ax.hist(iid, bins=bins, color=PALETTE[0], histtype="step", lw=2, label="Null: iid noise")
  ax.hist(ar1, bins=bins, color=PALETTE[1], histtype="step", lw=2,
          label=f"Null: AR(1) noise, phi={phi:.2f}")
  ax.axvline(obs, color=INK_2, lw=2, ls="--",
             label=f"Observed R2 = {obs:.2f} (p = {res['p_value_ar1']:.2f}, AR(1) null)")
  ax.set_xlabel("Variance share explained by week-of-year means")
  ax.set_ylabel("Simulations")
  ax.set_title("Yearly seasonality vs chance (detrended national revenue)")
  ax.legend(loc="upper left", fontsize=8)
  res["figure"] = savefig(fig, "seasonality_null.png")
  return res


def noise_floor(df, nat):
  def noise_stats(Y):
    """Noise measures for each series (row) of Y: shape (series, weeks)."""
    Y = np.atleast_2d(Y)
    detrended = np.apply_along_axis(detrend, 1, Y)
    moving_avg = pd.DataFrame(Y.T).rolling(13, center=True).mean().to_numpy().T
    return {
        "cv": Y.std(1, ddof=1) / Y.mean(1),
        "lag1": np.array([np.corrcoef(x[:-1], x[1:])[0, 1] for x in detrended]),
        "mape_vs_ma": np.nanmean(np.abs(Y / moving_avg - 1), 1),
    }

  def spread(values):
    return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}

  geo = noise_stats(df.pivot(index=GEO, columns=TIME, values="revenue").to_numpy())
  national = noise_stats(nat.to_numpy())
  # Noise vs geo size (input to using one residual variance for all geos).
  pc = df.assign(pc=df["revenue"] / df[POPULATION]).groupby(GEO).agg(
      pop=(POPULATION, "first"), sd=("pc", "std"))
  return {
      "definition": ("CV = sd/mean of weekly revenue; lag-1 autocorrelation of detrended "
                     "revenue; MAPE of a 13-week centred moving average (the error floor of "
                     "a smooth predictor)."),
      "geo_weekly_cv": spread(geo["cv"]),
      "geo_lag1_autocorr": spread(geo["lag1"]),
      "geo_mape_vs_13wk_ma": spread(geo["mape_vs_ma"]),
      "national_weekly_cv": float(national["cv"][0]),
      "national_lag1_autocorr": float(national["lag1"][0]),
      "national_mape_vs_13wk_ma": float(national["mape_vs_ma"][0]),
      "per_capita_sd_spearman_with_population": float(stats.spearmanr(pc["pop"], pc["sd"])[0]),
      "per_capita_sd_min_over_max": float(pc["sd"].min() / pc["sd"].max()),
  }


# --- Media ---------------------------------------------------------------------
def cost_per_media_unit(eda):
  """CPM per channel, and Meridian's spend-share chart."""
  outcome = eda.eda_engine.check_cost_per_media_unit()
  a = outcome.analysis_artifacts[0]
  cpm = a.cost_per_media_unit_da * 1000  # cost per 1,000 impressions
  per_channel = {}
  for channel in cpm.channel.values:
    values = cpm.sel(channel=channel).values
    values = values[~np.isnan(values)]  # weeks with no impressions have no CPM
    per_channel[str(channel)] = {"min": float(values.min()), "median": float(np.median(values)),
                                 "max": float(values.max())}
  return {"severity": severity(outcome), "n_outliers_flagged": int(len(a.outlier_df)),
          "cpm": per_channel,
          "figure_spend_share": save_chart(eda.plot_relative_spend_share_barchart(geos=NATIONALIZE), "spend_share.png")}


def media_outliers(eda):
  """Unusual weeks in media and controls (Meridian counts them together)."""
  std = eda.eda_engine.check_std()
  return {"n_outliers_flagged": next(int(len(a.outlier_df)) for a in std.analysis_artifacts
                                     if a.variable == "treatment_control_scaled"),
          "figure": save_chart(eda.plot_treatments_without_non_media_boxplot(geos=NATIONALIZE),
                               "media_boxplot.png")}


def media_population(eda):
  """Rank correlation between population and each channel's raw spend, across geos."""
  raw = eda.eda_engine.check_population_corr_raw_media().analysis_artifacts[0].correlation_ds
  return {"spearman": {str(c): float(v) for c, v in zip(raw.media_channel.values, raw["media"].values)},
          "figure": save_chart(eda.plot_population_raw_media_correlation(), "population_media.png")}


def media_correlation(df):
  """Within-geo correlation between each channel's spend and revenue, across geo-weeks."""
  W = demean(df, SPEND_COLS + ["revenue"], GEO)          # within-geo values
  tab = W.corr().set_axis(CHANNELS + ["revenue"], axis=0).set_axis(CHANNELS + ["revenue"], axis=1)
  values = tab.loc[CHANNELS, CHANNELS].abs().to_numpy()
  np.fill_diagonal(values, 0)                            # ignore each channel's correlation with itself
  i, j = np.unravel_index(values.argmax(), values.shape)
  # Lower triangle without the diagonal: rows from the 2nd variable, columns up to the 2nd-last.
  shown = tab.where(np.tril(np.ones(tab.shape, bool), k=-1)).iloc[1:, :-1]
  fig, ax = plt.subplots(figsize=(5, 4))
  ax.imshow(shown.to_numpy(float), cmap=DIVERGING, vmin=-1, vmax=1)
  ax.set_xticks(range(shown.shape[1]), shown.columns, rotation=45, ha="right")
  ax.set_yticks(range(shown.shape[0]), shown.index)
  ax.grid(False)
  for r in range(shown.shape[0]):
    for c in range(r + 1):
      ax.text(c, r, f"{shown.iloc[r, c]:.2f}", ha="center", va="center", fontsize=8)
  ax.set_title("Within-geo correlation of spend and revenue")
  return {"within_geo_corr": tab.to_dict(orient="index"), "max_abs_corr_between_channels": float(values[i, j]),
          "max_pair": [CHANNELS[i], CHANNELS[j]], "figure": savefig(fig, "media_correlation.png")}


def lag_correlations(df):
  # Everything is "within geo": each geo's own average is removed first.
  controls = sm.add_constant(demean(df, CONTROLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA, GEO))

  def remove_controls(values):
    """What is left of a series after regressing it on the controls, Promo and organic."""
    return values - controls @ np.linalg.lstsq(controls, values, rcond=None)[0]

  def by_lag(revenue, spend):
    """Correlation of revenue this week with spend 0, 1, ..., MAX_LAG weeks earlier."""
    out = []
    for lag in range(MAX_LAG + 1):
      earlier_spend = spend.groupby(df[GEO]).shift(lag)
      ok = earlier_spend.notna()
      out.append(float(np.corrcoef(revenue[ok], earlier_spend[ok])[0, 1]))
    return out

  revenue = demean(df, ["revenue"], GEO)["revenue"]
  revenue_adjusted = remove_controls(revenue)
  out = {"geo_demeaned": {}, "geo_demeaned_partial_controls": {}}
  for c, sp in zip(CHANNELS, SPEND_COLS):
    spend = demean(df, [sp], GEO)[sp]
    out["geo_demeaned"][c] = by_lag(revenue, spend)
    out["geo_demeaned_partial_controls"][c] = by_lag(revenue_adjusted, remove_controls(spend))
  # Approximate 95% band for a zero correlation with n geo-weeks.
  out["null_95pct_band"] = float(1.96 / np.sqrt(len(df)))
  part = out["geo_demeaned_partial_controls"]
  out["lags_above_null_band_partial"] = {
      c: [l for l, v in enumerate(vs) if v > out["null_95pct_band"]] for c, vs in part.items()}

  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
  for ax, key, title in [(axes[0], "geo_demeaned", "Raw (geo averages removed)"),
                         (axes[1], "geo_demeaned_partial_controls", "After removing the controls")]:
    ax.axhspan(-out["null_95pct_band"], out["null_95pct_band"], color="#e1e0d9", alpha=.6,
               lw=0)
    for c in CHANNELS:
      ax.plot(range(MAX_LAG + 1), out[key][c], marker="o", ms=4,
              color=CHANNEL_COLOR[c], label=c)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Lag of spend (weeks)")
    ax.set_title(title)
  axes[0].set_ylabel("corr(revenue_t, spend_t-lag)")
  axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
  fig.suptitle("Within-geo spend-revenue correlation by lag (grey band: range expected by chance)",
               y=1.03)
  out["figure"] = savefig(fig, "lag_correlation.png")
  return out


def spend_vs_geo_max(df):
  out = {}
  times = np.sort(df[TIME].unique())
  last52 = df[TIME] > times[-53]
  fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), sharey=True)
  for ax, c, sp in zip(axes, CHANNELS, SPEND_COLS):
    mx = df.groupby(GEO)[sp].transform("max")  # each geo's highest weekly spend, on every row
    x = df[sp] / mx                              # each week's spend as a share of that maximum
    nz = x[df[sp] > 0]                           # weeks when the channel was on
    zero_by_geo = (df[sp] == 0).groupby(df[GEO]).mean()  # share of off weeks, per geo
    out[c] = {
        "share_zero_geo_weeks": float((df[sp] == 0).mean()),
        "share_zero_by_geo_min": float(zero_by_geo.min()),
        "share_zero_by_geo_max": float(zero_by_geo.max()),
        "nonzero_ratio_to_geo_max_quantiles": {
            str(p): float(nz.quantile(p)) for p in (.1, .25, .5, .75, .9)},
        "share_nonzero_above_80pct_of_geo_max": float((nz > .8).mean()),
        # If last year's spend were raised by 20% (or 50%), how often would a geo-week go
        # beyond anything that geo has ever spent?
        "last52_share_geo_weeks_beyond_geo_max_if_x1.2": float(
            (x[last52] * 1.2 > 1).mean()),
        "last52_share_geo_weeks_beyond_geo_max_if_x1.5": float(
            (x[last52] * 1.5 > 1).mean()),
    }
    ax.hist(nz, bins=np.linspace(0, 1, 26), color=CHANNEL_COLOR[c])
    ax.set_title(f"{c}\nzero weeks {out[c]['share_zero_geo_weeks']:.0%}")
    ax.set_xlabel("Spend / geo max")
  axes[0].set_ylabel("Non-zero geo-weeks")
  fig.suptitle("Where geo-week spend sits relative to each geo's maximum", y=1.12)
  out["figure"] = savefig(fig, "spend_vs_geo_max.png")
  return out


# --- Control variables ---------------------------------------------------------
def controls_spread(eda):
  """Box plot of the controls and Promo (their unusual weeks are counted in media_outliers)."""
  return {"figure": save_chart(eda.plot_controls_and_non_media_boxplot(geos=NATIONALIZE),
                               "controls_boxplot.png")}


def controls_confounding(df):
  cols = CONTROLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA
  W = demean(df, cols + SPEND_COLS + ["revenue"], GEO)   # within-geo values
  # One row per control, with its correlation with revenue and with each channel's spend.
  tab = pd.DataFrame(index=cols, columns=["revenue"] + CHANNELS, dtype=float)
  for c in cols:
    tab.loc[c, "revenue"] = W[c].corr(W["revenue"])
    for ch, sp in zip(CHANNELS, SPEND_COLS):
      tab.loc[c, ch] = W[c].corr(W[sp])
  fig, ax = plt.subplots(figsize=(7, 2.6))
  ax.imshow(tab.to_numpy(float), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
  ax.set_xticks(range(tab.shape[1]), tab.columns)
  ax.set_yticks(range(tab.shape[0]), tab.index)
  ax.grid(False)
  for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
      ax.text(j, i, f"{tab.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
  ax.set_title("Within-geo correlation of controls/treatments with revenue and spend")
  return {
      "within_geo_corr": tab.to_dict(orient="index"),
      "max_abs_corr_control_with_spend_within_geo": float(
          tab.loc[CONTROLS, CHANNELS].abs().to_numpy().max()),
      "figure": savefig(fig, "controls_correlation.png"),
  }


def main():
  df = load_raw()
  nat = national_series(df)
  eda = meridian_eda_setup()
  summary = {
      # The data
      "data": sufficiency(df),
      "summary_stats": summary_stats(df),
      "data_adequacy": data_adequacy(eda),
      # Revenue
      "revenue_outliers": revenue_outliers(eda),
      "trend": trend(nat),
      "seasonality": seasonality_test(df, nat),
      "noise_floor": noise_floor(df, nat),
      # Media
      "cost_per_media_unit": cost_per_media_unit(eda),
      "media_outliers": media_outliers(eda),
      "media_population": media_population(eda),
      "media_correlation": media_correlation(df),
      "lag_correlation": lag_correlations(df),
      "spend_vs_geo_max": spend_vs_geo_max(df),
      # Control variables
      "controls_spread": controls_spread(eda),
      "controls_confounding": controls_confounding(df),
  }
  write_json(summary, OUT / "eda_summary.json")
  print(f"Wrote {OUT / 'eda_summary.json'}")


if __name__ == "__main__":
  main()
