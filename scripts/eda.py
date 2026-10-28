"""Extended EDA (`make eda`): summary JSON and figures.

Data checks and exploratory tests that inform the model specification. Every
EDA number quoted in the README is read from outputs/eda/eda_summary.json
produced here.

Revenue is conversions x revenue_per_conversion (the modelled KPI is
conversions, with revenue_per_conversion turning it into revenue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (CHANNEL_COLOR, CHANNELS, CONTROLS, FIGURES, GEO, INK_2,
                    KPI, MUTED, NON_MEDIA_TREATMENTS, ORGANIC_MEDIA, OUTPUTS,
                    PALETTE, POPULATION, REVENUE_PER_KPI, SEED, SPEND_COLS,
                    IMPRESSION_COLS, TIME, load_raw, style_matplotlib,
                    write_json)

style_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG = FIGURES / "eda"
FIG.mkdir(parents=True, exist_ok=True)
OUT = OUTPUTS / "eda"
N_SIM = 10_000
MAX_LAG = 8
SEASON = 52
# Newey-West bandwidth rule floor(4 (T/100)^(2/9)).
def nw_lags(n): return int(np.floor(4 * (n / 100) ** (2 / 9)))

DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#eb6834", "#f3f2ee", "#2a78d6"])


def savefig(fig, name):
  fig.savefig(FIG / name)
  plt.close(fig)
  return f"reports/figures/eda/{name}"


def demean(df, cols, by):
  return df[cols] - df.groupby(by)[cols].transform("mean")


# --------------------------------------------------------------------------
def sufficiency(df):
  geos = df[GEO].nunique()
  times = np.sort(df[TIME].unique())
  steps = np.unique(np.diff(times).astype("timedelta64[D]").astype(int))
  per_geo = df.groupby(GEO).size()
  num = df.select_dtypes("number")
  spend_total = df[SPEND_COLS].sum()
  share = spend_total / spend_total.sum()
  last52 = df[df[TIME] > times[-53]]
  nonzero_weeks = {c: int((df[s] > 0).sum()) for c, s in zip(CHANNELS, SPEND_COLS)}
  rpk = df[REVENUE_PER_KPI]
  return {
      "n_rows": len(df), "n_geos": geos, "n_weeks": len(times),
      "first_week": str(pd.Timestamp(times[0]).date()),
      "last_week": str(pd.Timestamp(times[-1]).date()),
      "time_step_days": steps.tolist(),
      "rows_per_geo_min": int(per_geo.min()), "rows_per_geo_max": int(per_geo.max()),
      "balanced_panel": bool(per_geo.nunique() == 1 and len(df) == geos * len(times)),
      "duplicate_geo_weeks": int(df.duplicated([GEO, TIME]).sum()),
      "missing_values_total": int(df.isna().sum().sum()),
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


def cpm_check(df):
  out, rows = {}, []
  for c, imp, sp in zip(CHANNELS, IMPRESSION_COLS, SPEND_COLS):
    d = df[[GEO, TIME, imp, sp]]
    pos = d[d[imp] > 0]
    cpm = 1000 * pos[sp] / pos[imp]
    med = cpm.median()
    fit = sm.OLS(d[sp], d[[imp]]).fit()
    geo_med = (1000 * pos[sp] / pos[imp]).groupby(pos[GEO]).median()
    nat = d.groupby(TIME)[[imp, sp]].sum()
    nat_cpm = 1000 * nat[sp] / nat[imp].replace(0, np.nan)
    out[c] = {
        "cpm_median": float(med), "cpm_cv": float(cpm.std() / cpm.mean()),
        "cpm_p01": float(cpm.quantile(.01)), "cpm_p99": float(cpm.quantile(.99)),
        "cpm_min": float(cpm.min()), "cpm_max": float(cpm.max()),
        "share_within_1pct_of_median": float((abs(cpm / med - 1) < .01).mean()),
        "max_abs_relative_deviation_from_median": float((cpm / med - 1).abs().max()),
        "spend_on_impressions_r2_no_intercept": float(fit.rsquared),
        "geo_median_cpm_cv": float(geo_med.std() / geo_med.mean()),
        "national_weekly_cpm_cv": float(nat_cpm.std() / nat_cpm.mean()),
        "rows_spend_without_impressions": int(((d[sp] > 0) & (d[imp] == 0)).sum()),
        "rows_impressions_without_spend": int(((d[sp] == 0) & (d[imp] > 0)).sum()),
    }
    rows.append(pd.DataFrame({"channel": c, "ppm": 1e6 * (cpm / med - 1)}))
  # Constant CPM if every geo-week is within float32 rounding of the median.
  out["constant_cpm_verdict"] = {
      c: bool(out[c]["max_abs_relative_deviation_from_median"] < 1e-6) for c in CHANNELS}
  out["note"] = ("Deviations are at float32 rounding level, so spend = impressions x a "
                 "constant CPM; any apparent CPM variation is rounding.")

  allr = pd.concat(rows)
  fig, ax = plt.subplots(figsize=(7, 3.2))
  for i, c in enumerate(CHANNELS):
    v = allr.loc[allr.channel == c, "ppm"].to_numpy()
    rng = np.random.default_rng(i)
    ax.scatter(i + rng.uniform(-.28, .28, len(v)), v, s=4, alpha=.25,
               color=CHANNEL_COLOR[c], linewidths=0)
  ax.axhline(0, color=MUTED, lw=1, ls="--")
  ax.set_xticks(range(len(CHANNELS)),
                [f"{c}\nCPM {out[c]['cpm_median']:.2f}" for c in CHANNELS])
  ax.set_ylabel("Deviation from median CPM\n(parts per million)")
  worst = max(out[c]["max_abs_relative_deviation_from_median"] for c in CHANNELS)
  ax.set_title(f"CPM is constant per channel: max deviation {worst * 1e6:.2f} ppm "
               "(float rounding)")
  out["figure"] = savefig(fig, "cpm_relative.png")
  return out


def national_series(df):
  return df.groupby(TIME)["revenue"].sum()


def trend_test(nat):
  y = nat.to_numpy()
  n = len(y)
  t = np.arange(n) / 52.0
  X = sm.add_constant(t)
  fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(n)})
  res = {"slope_per_year_pct_of_mean": float(100 * fit.params[1] / y.mean()),
         "t": float(fit.tvalues[1]), "p_value": float(fit.pvalues[1]),
         "hac_maxlags": nw_lags(n)}
  # Is a straight line enough? Add yearly hinge terms (a piecewise-linear trend
  # with breaks at weeks 52 and 104) and test them jointly with HAC SEs.
  tw = np.arange(n)
  Xh = np.column_stack([X, np.maximum(0, tw - 52) / 52.0, np.maximum(0, tw - 104) / 52.0])
  fit_h = sm.OLS(y, Xh).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(n)})
  wald = fit_h.wald_test(np.eye(4)[2:], scalar=True)
  res["piecewise_yearly_vs_linear"] = {
      "wald_stat": float(wald.statistic), "df": 2, "p_value": float(wald.pvalue),
      "linear_rejected_5pct": bool(wald.pvalue < 0.05)}
  res["first_52_vs_last_52_revenue_pct"] = float(100 * (y[-52:].sum() / y[:52].sum() - 1))

  fig, ax = plt.subplots(figsize=(8, 3.2))
  ax.plot(nat.index, y / 1e6, color=PALETTE[0], lw=1.2, label="National weekly revenue")
  ma = pd.Series(y).rolling(13, center=True).mean()
  ax.plot(nat.index, ma / 1e6, color=PALETTE[1], lw=2, label="13-week centred mean")
  fit = sm.OLS(y, X).fit()
  ax.plot(nat.index, fit.fittedvalues / 1e6, color=INK_2, lw=1.2, ls="--",
          label="Linear trend")
  ax.set_ylabel("Revenue (millions)")
  ax.set_title("National revenue: trend")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
  res["figure"] = savefig(fig, "national_trend.png")
  return res


def woy_r2(Y, period=SEASON):
  """Share of variance explained by week-of-year means. Y: (..., T)."""
  T = Y.shape[-1]
  g = np.arange(T) % period
  Yc = Y - Y.mean(-1, keepdims=True)
  means = np.stack([Yc[..., g == k].mean(-1) for k in range(period)], -1)
  fitted = means[..., g]
  return (fitted ** 2).sum(-1) / (Yc ** 2).sum(-1)


def detrend(y):
  t = np.arange(len(y))
  b = np.polyfit(t, y, 1)
  return y - np.polyval(b, t)


def seasonality_test(df, nat):
  rng = np.random.default_rng(SEED)
  y = detrend(nat.to_numpy())
  T = len(y)
  obs = float(woy_r2(y))
  phi = float(np.corrcoef(y[:-1], y[1:])[0, 1])
  iid = woy_r2(rng.standard_normal((N_SIM, T)))
  e = rng.standard_normal((N_SIM, T))
  ar = np.empty_like(e)
  ar[:, 0] = e[:, 0] / np.sqrt(1 - phi ** 2)
  for t in range(1, T):
    ar[:, t] = phi * ar[:, t - 1] + e[:, t]
  ar1 = woy_r2(ar)

  # Per geo: detrended revenue, iid null (fraction of geos significant at 5%).
  geo_y = df.pivot(index=GEO, columns=TIME, values="revenue").to_numpy()
  geo_r2 = woy_r2(np.apply_along_axis(detrend, 1, geo_y))
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
  piv = df.pivot(index=TIME, columns=GEO, values="revenue")
  def stats_for(Y):
    Y = np.atleast_2d(Y)
    cv = Y.std(1, ddof=1) / Y.mean(1)
    det = np.apply_along_axis(detrend, 1, Y)
    ac1 = np.array([np.corrcoef(x[:-1], x[1:])[0, 1] for x in det])
    ma = pd.DataFrame(Y.T).rolling(13, center=True).mean().to_numpy().T
    ape = np.nanmean(np.abs(Y / ma - 1), 1)
    return cv, ac1, ape
  g = stats_for(piv.to_numpy().T)
  n = stats_for(nat.to_numpy())
  q = lambda a: {"median": float(np.median(a)), "min": float(np.min(a)),
                 "max": float(np.max(a))}
  # Noise vs region size (input to using one residual variance for all regions).
  pc = df.assign(pc=df["revenue"] / df[POPULATION]).groupby(GEO).agg(
      pop=(POPULATION, "first"), sd=("pc", "std"))
  return {
      "definition": ("CV = sd/mean of weekly revenue; lag-1 autocorrelation of detrended "
                     "revenue; MAPE of a 13-week centred moving average (the error floor of "
                     "a smooth predictor)."),
      "geo_weekly_cv": q(g[0]), "geo_lag1_autocorr": q(g[1]), "geo_mape_vs_13wk_ma": q(g[2]),
      "national_weekly_cv": float(n[0][0]), "national_lag1_autocorr": float(n[1][0]),
      "national_mape_vs_13wk_ma": float(n[2][0]),
      "per_capita_sd_spearman_with_population": float(stats.spearmanr(pc["pop"], pc["sd"])[0]),
      "per_capita_sd_min_over_max": float(pc["sd"].min() / pc["sd"].max()),
  }


def lag_correlations(df):
  d = df.copy()
  d["rev_g"] = demean(d, ["revenue"], GEO)
  # Partial version: geo-demeaned revenue and spend with controls, Promo and
  # organic (geo-demeaned) regressed out of both.
  Z = sm.add_constant(demean(d, CONTROLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA, GEO))
  resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
  d["rev_p"] = resid(d["rev_g"])
  out = {"geo_demeaned": {}, "geo_demeaned_partial_controls": {}}
  for c, sp in zip(CHANNELS, SPEND_COLS):
    s_g = demean(d, [sp], GEO)[sp]
    s_p = resid(s_g)
    for key, rv, sv in [("geo_demeaned", d["rev_g"], s_g),
                        ("geo_demeaned_partial_controls", d["rev_p"], s_p)]:
      vals = []
      for lag in range(MAX_LAG + 1):
        lagged = sv.groupby(d[GEO]).shift(lag)
        m = lagged.notna()
        vals.append(float(np.corrcoef(rv[m], lagged[m])[0, 1]))
      out[key][c] = vals
  # Approximate 95% band for a zero correlation with n geo-weeks.
  out["null_95pct_band"] = float(1.96 / np.sqrt(len(d)))
  part = out["geo_demeaned_partial_controls"]
  out["lags_above_null_band_partial"] = {
      c: [l for l, v in enumerate(vs) if v > out["null_95pct_band"]] for c, vs in part.items()}

  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
  for ax, key, title in [(axes[0], "geo_demeaned", "Raw (region averages removed)"),
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
  fig.suptitle("Within-region spend-revenue correlation by lag (grey band: range expected by chance)",
               y=1.03)
  out["figure"] = savefig(fig, "lag_correlation.png")
  return out


def collinearity(df):
  cols = SPEND_COLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA + CONTROLS
  W = demean(df, cols, GEO)
  corr = W[SPEND_COLS].corr()
  corr.index = corr.columns = CHANNELS
  vif = {c: float(variance_inflation_factor(W[cols].to_numpy(), i)) for i, c in enumerate(cols)}
  off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()

  fig, ax = plt.subplots(figsize=(4.8, 3.8))
  ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
  ax.set_xticks(range(5), CHANNELS, rotation=45, ha="right")
  ax.set_yticks(range(5), CHANNELS)
  ax.grid(False)
  for i in range(5):
    for j in range(5):
      ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
  ax.set_title("Within-region spend correlation")
  return {"within_geo_spend_corr": corr.to_dict(), "within_geo_max_abs_corr": float(off.max()),
          "vif_within_geo": vif, "max_vif": float(max(vif.values())),
          "figure": savefig(fig, "collinearity.png")}


def population_relation(df):
  g = df.groupby(GEO).agg(population=(POPULATION, "first"), revenue=("revenue", "sum"),
                          **{c: (s, "sum") for c, s in zip(CHANNELS, SPEND_COLS)})
  g["spend"] = g[CHANNELS].sum(1)
  out = {}
  for col in ["revenue", "spend"] + CHANNELS:
    x, y = np.log(g["population"]), np.log(g[col].clip(lower=1e-9))
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    out[col] = {"pearson": float(stats.pearsonr(g["population"], g[col])[0]),
                "spearman": float(stats.spearmanr(g["population"], g[col])[0]),
                "loglog_slope": float(fit.params.iloc[1]),
                "loglog_slope_se": float(fit.bse.iloc[1])}
  pc = g["revenue"] / g["population"]
  out["revenue_per_capita_cv_across_geos"] = float(pc.std() / pc.mean())
  out["spend_per_capita_cv_across_geos"] = float(
      (g["spend"] / g["population"]).std() / (g["spend"] / g["population"]).mean())
  ctrl = df.groupby(GEO)[CONTROLS + NON_MEDIA_TREATMENTS].mean()
  out["control_geo_mean_spearman_with_population"] = {
      c: float(stats.spearmanr(g["population"], ctrl[c])[0]) for c in ctrl.columns}
  out["population_min"] = float(g["population"].min())
  out["population_max"] = float(g["population"].max())

  fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
  for ax, col, title in [(axes[0], "revenue", "Revenue vs population"),
                         (axes[1], "spend", "Paid spend vs population")]:
    ax.scatter(g["population"] / 1e3, g[col] / 1e6, s=16, color=PALETTE[0])
    ax.set_xlabel("Population (thousands)")
    ax.set_ylabel(f"Total {col} over 156 weeks (millions)")
    ax.set_title(f"{title}\nlog-log slope {out[col]['loglog_slope']:.2f}")
  out["figure"] = savefig(fig, "population.png")
  return out


def spend_vs_geo_max(df):
  out = {}
  times = np.sort(df[TIME].unique())
  last52 = df[TIME] > times[-53]
  fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), sharey=True)
  for ax, c, sp in zip(axes, CHANNELS, SPEND_COLS):
    mx = df.groupby(GEO)[sp].transform("max")
    x = df[sp] / mx
    nz = x[df[sp] > 0]
    zero_by_geo = (df[sp] == 0).groupby(df[GEO]).mean()
    out[c] = {
        "share_zero_geo_weeks": float((df[sp] == 0).mean()),
        "share_zero_by_geo_min": float(zero_by_geo.min()),
        "share_zero_by_geo_max": float(zero_by_geo.max()),
        "nonzero_ratio_to_geo_max_quantiles": {
            str(p): float(nz.quantile(p)) for p in (.1, .25, .5, .75, .9)},
        "share_nonzero_above_80pct_of_geo_max": float((nz > .8).mean()),
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


def controls_confounding(df):
  cols = CONTROLS + NON_MEDIA_TREATMENTS + ORGANIC_MEDIA
  W = demean(df, cols + SPEND_COLS + ["revenue"], GEO)
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


def spend_share_fig(suff):
  share = suff["spend_share"]
  fig, ax = plt.subplots(figsize=(6, 2.4))
  for i, c in enumerate(CHANNELS):
    ax.barh(i, share[c], color=CHANNEL_COLOR[c], height=.6)
    ax.text(share[c] + .005, i, f"{share[c]:.1%}", va="center", color=INK_2)
  ax.set_yticks(range(5), CHANNELS)
  ax.invert_yaxis()
  ax.set_xlabel("Share of total paid spend (156 weeks)")
  ax.set_xlim(0, max(share.values()) * 1.2)
  ax.set_title("Spend share by channel")
  return savefig(fig, "spend_share.png")


def main():
  df = load_raw()
  nat = national_series(df)
  summary = {"data": sufficiency(df)}
  summary["data"]["figure_spend_share"] = spend_share_fig(summary["data"])
  summary["cpm"] = cpm_check(df)
  summary["trend"] = trend_test(nat)
  summary["seasonality"] = seasonality_test(df, nat)
  summary["noise_floor"] = noise_floor(df, nat)
  summary["lag_correlation"] = lag_correlations(df)
  summary["collinearity"] = collinearity(df)
  summary["population"] = population_relation(df)
  summary["spend_vs_geo_max"] = spend_vs_geo_max(df)
  summary["controls_confounding"] = controls_confounding(df)
  write_json(summary, OUT / "eda_summary.json")
  print(f"Wrote {OUT / 'eda_summary.json'}")


if __name__ == "__main__":
  main()
