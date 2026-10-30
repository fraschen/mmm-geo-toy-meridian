"""Phase 6 inputs: posterior quantities for the reports (production model).

Writes outputs/05_report/:
  national_weekly.csv     observed, expected (mean + 94% predictive interval), baseline
  weekly_contribution.csv posterior-mean weekly incremental revenue by paid channel
  contributions.json      total baseline / channel / non-paid contributions, and how much
                          the baseline trades off with media
  roi_prior_posterior.csv prior and posterior ROI summary per channel
  hill_curves.csv         Meridian Analyzer.hill_curves (prior + posterior, 94%)
  response_curves.csv     Meridian Analyzer.response_curves (posterior, 94%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CI, CHANNELS, GEO, OUTPUTS, TIME, summarize, load_model, load_raw, revenue_draws, write_json

OUT = OUTPUTS / "05_report"


def weekly(mmm, an, obs, times):
  """National weekly series, and per-week contributions (chain, draw, week, ...) for the totals."""
  mean, pred = revenue_draws(mmm)
  expected = mean.sum(2)                                   # national: sum over geos
  # Incremental revenue per week and channel: paid channels first, then organic, then Promo.
  inc = np.asarray(an.incremental_outcome(use_kpi=False, aggregate_times=False,
                                          include_non_paid_channels=True))
  paid, nonpaid = inc[..., :len(CHANNELS)], inc[..., len(CHANNELS):]
  baseline = expected - inc.sum(-1)                        # what is left without any driver
  lo, hi = np.quantile(pred.sum(2), CI, axis=(0, 1))
  pd.DataFrame({"time": times, "observed": obs.sum(0), "expected": expected.mean((0, 1)),
                "pred_lo": lo, "pred_hi": hi, "baseline": baseline.mean((0, 1))}
               ).to_csv(OUT / "national_weekly.csv", index=False)
  contribution = pd.DataFrame(paid.mean((0, 1)), columns=CHANNELS)
  contribution.insert(0, "time", times)
  contribution.to_csv(OUT / "weekly_contribution.csv", index=False)
  return baseline, paid, nonpaid


def contributions(mmm, obs, baseline, paid, nonpaid):
  """Totals over all weeks, one value per posterior draw, summarised."""
  base_tot = baseline.sum(-1).ravel()
  paid_tot = paid.sum(2).reshape(-1, len(CHANNELS))
  nonpaid_tot = nonpaid.sum(2).reshape(-1, nonpaid.shape[-1])
  nonpaid_names = ([str(c) for c in np.asarray(mmm.input_data.organic_media_channel)]
                   + [str(c) for c in np.asarray(mmm.input_data.non_media_channel)])
  write_json({"observed_total": float(obs.sum()),
              "baseline": summarize(base_tot),
              "paid": {c: summarize(paid_tot[:, i]) for i, c in enumerate(CHANNELS)},
              "paid_total": summarize(paid_tot.sum(1)),
              "non_paid": {c: summarize(nonpaid_tot[:, i]) for i, c in enumerate(nonpaid_names)},
              # Identification: across draws, does a higher baseline come with less media?
              "corr_baseline_vs_paid_media": float(np.corrcoef(base_tot, paid_tot.sum(1))[0, 1])},
             OUT / "contributions.json")


def roi_prior_posterior(an):
  post = np.asarray(an.roi(use_kpi=False)).reshape(-1, len(CHANNELS))
  prior = np.asarray(an.roi(use_posterior=False, use_kpi=False)).reshape(-1, len(CHANNELS))
  rows = []
  for i, c in enumerate(CHANNELS):
    for label, x in (("prior", prior[:, i]), ("posterior", post[:, i])):
      rows.append({"channel": c, "distribution": label, "mean": x.mean(), "sd": x.std(),
                   "ci_lo": np.quantile(x, CI[0]), "ci_hi": np.quantile(x, CI[1]),
                   "p_roi_gt_1": (x > 1).mean()})
  pd.DataFrame(rows).to_csv(OUT / "roi_prior_posterior.csv", index=False)


def curves(an):
  an.hill_curves(confidence_level=0.94).to_csv(OUT / "hill_curves.csv", index=False)
  multipliers = list(np.round(np.arange(0, 2.01, 0.05), 2))   # 0%, 5%, ..., 200% of historical spend
  (an.response_curves(spend_multipliers=multipliers, confidence_level=0.94, use_kpi=False)
   .to_dataframe().reset_index().to_csv(OUT / "response_curves.csv", index=False))


def main():
  from meridian.analysis.analyzer import Analyzer
  OUT.mkdir(parents=True, exist_ok=True)
  mmm = load_model()
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  times = [str(t) for t in np.asarray(mmm.input_data.time)]
  geos = list(np.asarray(mmm.input_data.geo))
  obs = load_raw().pivot(index=GEO, columns=TIME, values="revenue").loc[geos].to_numpy()  # geos x weeks

  baseline, paid, nonpaid = weekly(mmm, an, obs, times)
  contributions(mmm, obs, baseline, paid, nonpaid)
  roi_prior_posterior(an)
  curves(an)
  print(f"Wrote {OUT}")


if __name__ == "__main__":
  main()
