"""Phase 6 inputs: posterior quantities for the reports (selected model only).

Writes outputs/05_report/:
  national_weekly.csv     observed, expected (mean + 94% predictive interval), baseline
  weekly_contribution.csv posterior-mean weekly incremental revenue by paid channel
  contributions.json      total baseline / channel / non-paid contributions
  hill_curves.csv         Meridian Analyzer.hill_curves (prior + posterior, 94%)
  response_curves.csv     Meridian Analyzer.response_curves (posterior, 94%)
  roi_prior_posterior.csv prior and posterior ROI draws summary per channel
  identification.json     posterior correlation of baseline level with media contribution
  calibration_levels.json national / geo-week coverage at 50/80/94% (in-sample)
  rhat_by_variable.csv    max R-hat / min ESS per variable
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (CHANNELS, GEO, OUTPUTS, TIME, load_model, load_raw, revenue_draws,
                    write_json)

CI = (0.03, 0.97)
OUT = OUTPUTS / "05_report"


def main():
  from meridian.analysis.analyzer import Analyzer
  OUT.mkdir(parents=True, exist_ok=True)
  mmm = load_model()
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  times = [str(t) for t in np.asarray(mmm.input_data.time)]
  geos = list(np.asarray(mmm.input_data.geo))
  raw = load_raw()
  obs = raw.pivot(index=GEO, columns=TIME, values="revenue").loc[geos].to_numpy()

  # Expected outcome and incremental contributions by week (national).
  mean, pred = revenue_draws(mmm)
  exp_nat = mean.sum(2)  # (chain, draw, time)
  inc_all = np.asarray(an.incremental_outcome(use_kpi=False, aggregate_times=False,
                                              include_non_paid_channels=True))
  # (chain, draw, time, channel): paid channels first, then organic, then non-media.
  n = len(CHANNELS)
  paid_w = inc_all[..., :n]
  nonpaid_w = inc_all[..., n:]
  baseline_w = exp_nat - inc_all.sum(-1)
  lo, hi = np.quantile(pred.sum(2), CI, axis=(0, 1))
  pd.DataFrame({"time": times, "observed": obs.sum(0), "expected": exp_nat.mean((0, 1)),
                "pred_lo": lo, "pred_hi": hi, "baseline": baseline_w.mean((0, 1)),
                "baseline_lo": np.quantile(baseline_w, CI[0], axis=(0, 1)),
                "baseline_hi": np.quantile(baseline_w, CI[1], axis=(0, 1))}
               ).to_csv(OUT / "national_weekly.csv", index=False)
  wc = pd.DataFrame(paid_w.mean((0, 1)), columns=CHANNELS)
  wc.insert(0, "time", times)
  wc.to_csv(OUT / "weekly_contribution.csv", index=False)

  tot = lambda x: x.sum(2).ravel() if x.ndim == 3 else x.sum(2).reshape(-1, x.shape[-1])
  paid_tot = paid_w.sum(2).reshape(-1, n)
  nonpaid_tot = nonpaid_w.sum(2).reshape(-1, nonpaid_w.shape[-1])
  base_tot = baseline_w.sum(-1).ravel()
  s = lambda x: {"mean": float(x.mean()), "ci_lo": float(np.quantile(x, CI[0])),
                 "ci_hi": float(np.quantile(x, CI[1]))}
  nonpaid_names = [str(c) for c in np.asarray(mmm.input_data.organic_media_channel)] + \
      [str(c) for c in np.asarray(mmm.input_data.non_media_channel)]
  contrib = {"observed_total": float(obs.sum()), "expected_total": s(exp_nat.sum(-1).ravel()),
             "baseline": s(base_tot),
             "paid": {c: s(paid_tot[:, i]) for i, c in enumerate(CHANNELS)},
             "paid_total": s(paid_tot.sum(1)),
             "non_paid": {c: s(nonpaid_tot[:, i]) for i, c in enumerate(nonpaid_names)}}
  write_json(contrib, OUT / "contributions.json")

  # Identification: does the baseline level trade off with media?
  media_tot = paid_tot.sum(1)
  write_json({"corr_baseline_total_vs_paid_media_total": float(np.corrcoef(base_tot, media_tot)[0, 1]),
              "corr_by_channel": {c: float(np.corrcoef(base_tot, paid_tot[:, i])[0, 1])
                                  for i, c in enumerate(CHANNELS)}},
             OUT / "identification.json")

  # ROI prior vs posterior.
  post = np.asarray(an.roi(use_kpi=False)).reshape(-1, n)
  prior = np.asarray(an.roi(use_posterior=False, use_kpi=False)).reshape(-1, n)
  rows = []
  for i, c in enumerate(CHANNELS):
    for lab, x in (("prior", prior[:, i]), ("posterior", post[:, i])):
      rows.append({"channel": c, "distribution": lab, "mean": x.mean(), "sd": x.std(),
                   "ci_lo": np.quantile(x, CI[0]), "ci_hi": np.quantile(x, CI[1]),
                   "p_roi_gt_1": (x > 1).mean()})
  pd.DataFrame(rows).to_csv(OUT / "roi_prior_posterior.csv", index=False)

  # Curves.
  an.hill_curves(confidence_level=0.94).to_csv(OUT / "hill_curves.csv", index=False)
  rc = an.response_curves(spend_multipliers=list(np.round(np.arange(0, 2.01, 0.05), 2)),
                          confidence_level=0.94, use_kpi=False)
  rcd = rc.to_dataframe().reset_index()
  rcd.to_csv(OUT / "response_curves.csv", index=False)

  # Calibration of predictive intervals at several levels (in-sample).
  levels = {}
  for lvl in (0.5, 0.8, 0.94):
    q = ((1 - lvl) / 2, 1 - (1 - lvl) / 2)
    glo, ghi = np.quantile(pred, q, axis=(0, 1))
    nlo, nhi = np.quantile(pred.sum(2), q, axis=(0, 1))
    levels[str(lvl)] = {"geo_week": float(np.mean((obs >= glo) & (obs <= ghi))),
                        "national": float(np.mean((obs.sum(0) >= nlo) & (obs.sum(0) <= nhi)))}
  write_json(levels, OUT / "calibration_levels.json")

  conv = json.load(open(OUTPUTS / "02_fit" / "convergence.json"))
  pd.DataFrame([{"variable": v, **p} for v, p in conv["per_variable"].items()]
               ).to_csv(OUT / "rhat_by_variable.csv", index=False)
  print(f"Wrote {OUT}")


if __name__ == "__main__":
  main()
