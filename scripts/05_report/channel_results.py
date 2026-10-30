"""Deliverable 2: national channel-level results (CSV + metadata JSON).

National level only (no geo-level ROI). Per channel: incremental revenue
(contribution), ROI and marginal ROI with 94% credible intervals, P(ROI > 1), spend,
spend share, share of revenue and the channel_roi_checks gate. The period, units and
column descriptions are in the metadata JSON.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import summarize, CHANNELS, OUTPUTS, load_model, load_model_config, load_raw, write_json

OUT = OUTPUTS / "05_report"


def main():
  from meridian.analysis.analyzer import Analyzer
  OUT.mkdir(parents=True, exist_ok=True)
  model_cfg = load_model_config()["model"]
  gates = json.load(open(OUTPUTS / "03_validate" / "gates.json"))
  channel_gate = next(g for g in gates["gates"] if g["gate"] == "channel_roi_checks")["value"]
  mmm = load_model()
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  n = len(CHANNELS)
  inc = np.asarray(an.incremental_outcome(use_kpi=False, include_non_paid_channels=False)).reshape(-1, n)
  roi = np.asarray(an.roi(use_kpi=False)).reshape(-1, n)
  mroi = np.asarray(an.marginal_roi(use_kpi=False)).reshape(-1, n)

  raw = load_raw()
  spend = raw[[f"{c}_spend" for c in CHANNELS]].sum().to_numpy()
  total_rev = float(raw["revenue"].sum())
  period = f"{raw['time'].min().date()} to {raw['time'].max().date()}"

  rows = []
  for i, c in enumerate(CHANNELS):
    contribution = summarize(inc[:, i])
    rows.append({
        "channel": c,
        **{f"contribution_{k}": v for k, v in contribution.items()},
        **{f"roi_{k}": v for k, v in summarize(roi[:, i]).items()},
        **{f"mroi_{k}": v for k, v in summarize(mroi[:, i]).items()},
        "p_roi_gt_1": float((roi[:, i] > 1).mean()),
        "spend": float(spend[i]), "spend_share": float(spend[i] / spend.sum()),
        "share_of_revenue": contribution["mean"] / total_rev,
        "gate_status": channel_gate[c]["status"],
        "prior_driven": channel_gate[c]["prior_driven"],
        "payback_determined": channel_gate[c]["payback_determined"],
    })
  df = pd.DataFrame(rows)
  df.to_csv(OUT / "channel_results.csv", index=False)

  meta = {
      "model": model_cfg["name"], "description": model_cfg["description"], "backend": "JAX",
      "period": period, "level": "national (sum over 40 geos)",
      "units": "revenue (conversions x revenue_per_conversion); ROI = revenue per unit spend",
      "credible_interval": "94% (3rd-97th percentile of 4 chains x 1000 posterior draws)",
      "columns": {
          "channel": "Paid media channel.",
          "contribution_mean/_ci_lo/_ci_hi": "Incremental revenue attributed to the channel (vs zero spend): posterior mean and 94% CI.",
          "roi_mean/_ci_lo/_ci_hi": "Incremental revenue / spend: posterior mean and 94% CI.",
          "mroi_mean/_ci_lo/_ci_hi": "Marginal ROI: incremental revenue from a 1% spend increase / that extra spend.",
          "p_roi_gt_1": "Posterior probability that ROI > 1.",
          "spend": "Total channel spend over the period.",
          "spend_share": "Channel spend / total paid spend.",
          "share_of_revenue": "Posterior mean contribution / total observed revenue.",
          "gate_status": "channel_roi_checks gate (PASS/WARN) from make gates.",
          "prior_driven": "True if the data barely narrowed the ROI (contraction below the gate minimum).",
          "payback_determined": "True if P(ROI > 1) is at least 0.9 or at most 0.1.",
      },
      "notes": ["Cost per incremental KPI is not reported: the modelled outcome is converted to revenue.",
                "No geo-level ROI is reported."],
  }
  write_json(meta, OUT / "channel_results_metadata.json")
  print(df[["channel", "roi_mean", "roi_ci_lo", "roi_ci_hi", "mroi_mean", "p_roi_gt_1",
            "spend_share", "share_of_revenue", "gate_status"]].round(3).to_string(index=False))


if __name__ == "__main__":
  main()
