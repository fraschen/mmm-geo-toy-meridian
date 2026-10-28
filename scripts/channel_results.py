"""Deliverable 2: national channel-level results (CSV + metadata JSON).

Selected model only, national level only (no geo-level ROI). Per channel:
incremental revenue (contribution), ROI and marginal ROI with 94% credible
intervals, P(ROI > 1), spend, spend share, share of revenue, ROI range across
converged CV folds, gate status, period and units.
"""

from __future__ import annotations

import importlib.metadata as md
import json

import numpy as np
import pandas as pd

from common import CHANNELS, OUTPUTS, load_model, load_model_config, load_raw, write_json

CI = (0.03, 0.97)
OUT = OUTPUTS / "results"


def stats(x):
  x = np.asarray(x).ravel()
  return float(x.mean()), float(np.quantile(x, CI[0])), float(np.quantile(x, CI[1]))


def main():
  from meridian.analysis.analyzer import Analyzer
  OUT.mkdir(parents=True, exist_ok=True)
  model_cfg = load_model_config()["model"]
  gates = json.load(open(OUTPUTS / "gates" / "gates.json"))
  name = model_cfg["name"]
  mmm = load_model()
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  n = len(CHANNELS)
  inc = np.asarray(an.incremental_outcome(use_kpi=False, include_non_paid_channels=False)).reshape(-1, n)
  roi = np.asarray(an.roi(use_kpi=False)).reshape(-1, n)
  mroi = np.asarray(an.marginal_roi(use_kpi=False)).reshape(-1, n)

  raw = load_raw()
  spend = raw[[f"{c}_spend" for c in CHANNELS]].sum().to_numpy()
  total_rev = float(raw["revenue"].sum())
  roi_cv = pd.read_csv(OUTPUTS / "validation" / "cv_roi.csv")
  n_conv_folds = roi_cv["fold"].nunique() if len(roi_cv) else 0
  ch_gate = next(g for g in gates["gates"] if g["gate"] == "channel_roi_checks")["value"]
  period = f"{raw['time'].min().date()} to {raw['time'].max().date()}"

  rows = []
  for i, c in enumerate(CHANNELS):
    cm, cl, ch = stats(inc[:, i])
    rm, rl, rh = stats(roi[:, i])
    mm, ml, mh = stats(mroi[:, i])
    folds = roi_cv[roi_cv.channel == c]["mean"] if n_conv_folds else pd.Series(dtype=float)
    g = ch_gate[c]
    rows.append({
        "channel": c, "period": period,
        "contribution_mean": cm, "contribution_ci_lo": cl, "contribution_ci_hi": ch,
        "roi_mean": rm, "roi_ci_lo": rl, "roi_ci_hi": rh,
        "mroi_mean": mm, "mroi_ci_lo": ml, "mroi_ci_hi": mh,
        "p_roi_gt_1": float((roi[:, i] > 1).mean()),
        "spend": float(spend[i]), "spend_share": float(spend[i] / spend.sum()),
        "share_of_revenue": cm / total_rev,
        "roi_cv_fold_min": float(folds.min()) if len(folds) >= 2 else None,
        "roi_cv_fold_max": float(folds.max()) if len(folds) >= 2 else None,
        "roi_cv_converged_folds": int(n_conv_folds),
        "gate_status": g["status"],
        "gate_notes": "; ".join(
            ([] if g["payback_determined"] else ["payback undetermined (0.1 < P(ROI>1) < 0.9)"])
            + (["prior-driven ROI"] if g["prior_driven"] else [])) or "none",
        "units": "revenue (conversions x revenue_per_conversion); ROI = revenue per unit spend",
    })
  df = pd.DataFrame(rows)
  df.to_csv(OUT / "channel_results.csv", index=False)

  versions = {p: md.version(p) for p in ("google-meridian", "jax", "jaxlib", "tfp-nightly",
                                         "numpy", "pandas", "arviz")}
  meta = {
      "model": name, "description": model_cfg["description"], "backend": "JAX",
      "period": period, "level": "national (sum over 40 geos)",
      "credible_interval": "94% (3rd-97th percentile of 4 chains x 1000 posterior draws)",
      "package_versions": versions,
      "columns": {
          "channel": "Paid media channel.",
          "period": "Weeks covered (all 156 modelled weeks).",
          "contribution_mean/_ci_lo/_ci_hi": "Incremental revenue attributed to the channel (vs zero spend): posterior mean and 94% CI.",
          "roi_mean/_ci_lo/_ci_hi": "Incremental revenue / spend: posterior mean and 94% CI.",
          "mroi_mean/_ci_lo/_ci_hi": "Marginal ROI: incremental revenue from a 1% spend increase / that extra spend.",
          "p_roi_gt_1": "Posterior probability that ROI > 1.",
          "spend": "Total channel spend over the period.",
          "spend_share": "Channel spend / total paid spend.",
          "share_of_revenue": "Posterior mean contribution / total observed revenue.",
          "roi_cv_fold_min/_max": "Range of fold-level posterior mean ROI across converged CV fold fits (null if fewer than 2 converged).",
          "roi_cv_converged_folds": "Number of CV fold fits that passed the convergence gate.",
          "gate_status": "Per-channel gate (PASS/WARN) from make gates.",
          "gate_notes": "Reason for WARN.",
          "units": "Units of monetary columns.",
      },
      "notes": ["Cost per incremental KPI is not reported: the modelled outcome is converted to revenue.",
                "No geo-level ROI is reported."],
  }
  write_json(meta, OUT / "channel_results_metadata.json")
  print(df[["channel", "roi_mean", "roi_ci_lo", "roi_ci_hi", "mroi_mean", "p_roi_gt_1",
            "spend_share", "share_of_revenue", "gate_status"]].round(3).to_string(index=False))


if __name__ == "__main__":
  main()
