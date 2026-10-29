"""Phase 5: automated gates for the selected spec (`make gates`).

Exits non-zero if any blocking gate fails. Writes outputs/gates/gates.json and
outputs/gates/insample_national.csv. Rules, thresholds and their basis are in
config/gates.toml; accuracy thresholds are set against the EDA noise floor
(eda_summary.json).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import (CHANNELS, GEO, OUTPUTS, TIME, load_raw, sha256, RAW_CSV,
                    DATA_SHA256, load_gates_config, load_model, load_model_config,
                    write_json)

CI = (0.03, 0.97)
OUT = OUTPUTS / "gates"

# Rules, thresholds and their basis: config/gates.toml.
CONFIG = load_gates_config()
THRESHOLDS = {k: v for g in CONFIG["gates"].values() for k, v in g["thresholds"].items()
              if g is not CONFIG["gates"]["convergence"]}
BASIS = {name: g["basis"] for name, g in CONFIG["gates"].items()}


def revenue_draws(mmm, use_posterior=True):
  """Expected and predictive revenue draws, (chain, draw, geo, time)."""
  from meridian import backend
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  mean_scaled = np.asarray(an.expected_outcome(
      use_posterior=use_posterior, use_kpi=True, aggregate_geos=False,
      aggregate_times=False, inverse_transform_outcome=False))
  group = mmm.inference_data.posterior if use_posterior else mmm.inference_data.prior
  sigma = np.asarray(group["sigma"])
  sigma = sigma[..., None, None] if sigma.ndim == 2 else sigma[..., None]
  rng = np.random.default_rng(1 if use_posterior else 2)
  pred_scaled = mean_scaled + sigma * rng.standard_normal(mean_scaled.shape)
  tr = mmm.model_context.kpi_transformer
  inv = lambda x: np.asarray(tr.inverse(backend.to_tensor(x, dtype=backend.float_dtype)))
  rpk = np.asarray(mmm.input_data.revenue_per_kpi)
  return inv(mean_scaled) * rpk, inv(pred_scaled) * rpk


def r2(obs, pred):
  return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def coverage(obs, draws):
  lo, hi = np.quantile(draws, CI, axis=(0, 1))
  return float(np.mean((obs >= lo) & (obs <= hi))), lo, hi


def lag1(x):
  x = x - x.mean()
  return float((x[:-1] * x[1:]).sum() / (x ** 2).sum())


def gate(name, status, value, threshold, blocking, note=""):
  return {"gate": name, "status": status, "value": value, "threshold": threshold,
          "blocking": blocking, "note": note}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  eda = json.load(open(OUTPUTS / "eda" / "eda_summary.json"))
  model_cfg = load_model_config()["model"]
  cv = json.load(open(OUTPUTS / "validation" / "cv_metrics.json"))
  health = json.load(open(OUTPUTS / "meridian" / "health_check.json"))
  name = model_cfg["name"]
  conv = json.load(open(OUTPUTS / "model" / "convergence.json"))
  nf = eda["noise_floor"]
  basis = {k: v.format(nat_floor=nf["national_mape_vs_13wk_ma"],
                       ratio=THRESHOLDS["cv_national_mape_max"] / nf["national_mape_vs_13wk_ma"],
                       geo_cv=nf["geo_weekly_cv"]["median"],
                       geo_floor=nf["geo_mape_vs_13wk_ma"]["median"],
                       nat_cv=nf["national_weekly_cv"], nat_ac=nf["national_lag1_autocorr"])
           for k, v in BASIS.items()}
  T = THRESHOLDS
  gates = []

  # 1. Data integrity.
  d = eda["data"]
  ok = (sha256(RAW_CSV) == DATA_SHA256 and d["balanced_panel"] and d["missing_values_total"] == 0
        and d["duplicate_geo_weeks"] == 0 and d["nonnegative_required_columns_ok"])
  gates.append(gate("data_integrity", "PASS" if ok else "FAIL",
                    {"checksum_ok": sha256(RAW_CSV) == DATA_SHA256,
                     "balanced_panel": d["balanced_panel"], "missing": d["missing_values_total"],
                     "duplicates": d["duplicate_geo_weeks"],
                     "nonnegative": d["nonnegative_required_columns_ok"]}, "all true", True))

  raw = load_raw()
  mmm = load_model()
  geos = list(np.asarray(mmm.input_data.geo))
  obs = raw.pivot(index=GEO, columns=TIME, values="revenue").loc[geos].to_numpy()

  # 2. Prior predictive coverage (national weekly revenue).
  _, prior_pred = revenue_draws(mmm, use_posterior=False)
  pcov, plo, phi = coverage(obs.sum(0), prior_pred.sum(2))
  gates.append(gate("prior_predictive_coverage", "PASS" if pcov >= T["prior_predictive_national_coverage_min"] else "FAIL",
                    pcov, T["prior_predictive_national_coverage_min"], True,
                    f"prior predictive 94% interval width / observed mean: "
                    f"{float(np.mean(phi - plo) / obs.sum(0).mean()):.2f}"))

  # 3. Convergence (config/gates.toml; outputs/model/convergence.json).
  th = conv["thresholds"]
  gates.append(gate("convergence", "PASS" if conv["converged"] else "FAIL",
                    {"max_rhat": conv["max_rhat"], "min_ess_bulk": conv["min_ess_bulk"],
                     "min_ess_tail": conv["min_ess_tail"], "divergences": conv["divergences"]},
                    {"rhat_max": th["rhat_max"], "ess_min": th["ess_min"],
                     "divergences_max": th["divergences_max"]}, True, th["note"]))

  # 4. In-sample fit.
  mean, pred = revenue_draws(mmm)
  m = mean.mean((0, 1))
  g_r2, n_r2 = r2(obs, m), r2(obs.sum(0), m.sum(0))
  g_cov, _, _ = coverage(obs, pred)
  n_cov, nlo, nhi = coverage(obs.sum(0), pred.sum(2))
  lo_c, hi_c = T["insample_coverage_range"]
  gates.append(gate("insample_r2", "PASS" if (g_r2 >= T["insample_geo_week_r2_min"] and n_r2 >= T["insample_national_r2_min"]) else "FAIL",
                    {"geo_week": g_r2, "national": n_r2},
                    {"geo_week_min": T["insample_geo_week_r2_min"], "national_min": T["insample_national_r2_min"]}, True))
  gates.append(gate("insample_coverage", "PASS" if (lo_c <= g_cov <= hi_c and lo_c <= n_cov <= hi_c) else "FAIL",
                    {"geo_week": g_cov, "national": n_cov}, T["insample_coverage_range"], True))
  insample = {"geo_week_mape": float(np.mean(np.abs(m / obs - 1))),
              "national_mape": float(np.mean(np.abs(m.sum(0) / obs.sum(0) - 1)))}

  # 5. Warnings: within-geo R2, residual autocorrelation.
  wo = obs - obs.mean(1, keepdims=True)
  wm = m - m.mean(1, keepdims=True)
  wg_r2 = r2(wo, wm)
  res = obs - m
  nat_ac = lag1(res.sum(0))
  geo_ac = float(np.median([lag1(r) for r in res]))
  gates.append(gate("within_geo_r2", "PASS" if wg_r2 >= T["warn_within_geo_r2_min"] else "WARN",
                    wg_r2, T["warn_within_geo_r2_min"], False))
  gates.append(gate("residual_autocorrelation",
                    "PASS" if max(abs(nat_ac), abs(geo_ac)) <= T["warn_residual_lag1_abs_max"] else "WARN",
                    {"national_lag1": nat_ac, "median_geo_lag1": geo_ac},
                    T["warn_residual_lag1_abs_max"], False))

  # 6. Holdout and each CV fold.
  for label, f in cv["folds"].items():
    if not f["converged"]:
      gates.append(gate(f"cv_{label}", "FAIL", {"converged": False},
                        "fold fit must converge", True,
                        "Fold fit did not converge; no forecast evidence (diagnostics only)."))
      continue
    gw, nat = f["geo_week"], f["national"]
    checks = {"geo_error_ratio": gw["error_ratio"] <= T["cv_error_ratio_max"],
              "national_error_ratio": nat["error_ratio"] <= T["cv_error_ratio_max"],
              "national_mape": nat["forecast"]["mape"] <= T["cv_national_mape_max"],
              "geo_coverage": gw["forecast"]["coverage_94"] >= T["cv_coverage_min"],
              "national_coverage": nat["forecast"]["coverage_94"] >= T["cv_coverage_min"]}
    gates.append(gate(f"cv_{label}", "PASS" if all(checks.values()) else "FAIL",
                      {"geo_error_ratio": gw["error_ratio"], "national_error_ratio": nat["error_ratio"],
                       "national_mape": nat["forecast"]["mape"],
                       "geo_coverage": gw["forecast"]["coverage_94"],
                       "national_coverage": nat["forecast"]["coverage_94"],
                       "national_bias": nat["forecast"]["mean_signed_pct_error"],
                       "failed": [k for k, v in checks.items() if not v]},
                      {"error_ratio_max": T["cv_error_ratio_max"],
                       "national_mape_max": T["cv_national_mape_max"],
                       "coverage_min": T["cv_coverage_min"]}, True))

  # 7. ROI stability across folds (converged folds only).
  roi_cv = pd.read_csv(OUTPUTS / "validation" / "cv_roi.csv")
  n_conv = roi_cv["fold"].nunique() if len(roi_cv) else 0
  if n_conv < 2:
    gates.append(gate("roi_fold_stability", "FAIL", {"converged_folds": int(n_conv)},
                      "at least 2 converged folds", True,
                      "Not assessable: fewer than two converged fold fits."))
    fold_ratio = {}
  else:
    piv = roi_cv.pivot(index="fold", columns="channel", values="mean")
    fold_ratio = (piv.max() / piv.min()).to_dict()
    worst = max(fold_ratio.values())
    status = "FAIL" if worst > T["roi_fold_ratio_fail"] else "WARN" if worst > T["roi_fold_ratio_warn"] else "PASS"
    gates.append(gate("roi_fold_stability", status, fold_ratio,
                      {"warn": T["roi_fold_ratio_warn"], "fail": T["roi_fold_ratio_fail"]}, True))

  # 8. Per channel: contraction vs prior and P(ROI > 1).
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  post = np.asarray(an.roi(use_kpi=False)).reshape(-1, len(CHANNELS))
  prior = np.asarray(an.roi(use_posterior=False, use_kpi=False)).reshape(-1, len(CHANNELS))
  lo_p, hi_p = T["p_roi_gt_1_determined"]
  channel_gates = {}
  for i, c in enumerate(CHANNELS):
    contraction = float(1 - post[:, i].std() / prior[:, i].std())
    p1 = float((post[:, i] > 1).mean())
    channel_gates[c] = {
        "roi_contraction": contraction,
        "prior_driven": contraction < T["roi_contraction_min"],
        "p_roi_gt_1": p1, "payback_determined": p1 >= hi_p or p1 <= lo_p,
        "status": "WARN" if (contraction < T["roi_contraction_min"] or lo_p < p1 < hi_p) else "PASS"}
  gates.append(gate("channel_roi_checks",
                    "WARN" if any(v["status"] == "WARN" for v in channel_gates.values()) else "PASS",
                    channel_gates, {"contraction_min": T["roi_contraction_min"],
                                    "p_roi_gt_1_determined": T["p_roi_gt_1_determined"]}, False))

  # 9. Meridian health checks (reported, not gating), with threshold comparison.
  looser = {"Convergence": (f"Meridian passes R-hat < {next(c['details'].get('convergence_threshold') for c in health['checks'] if c['check'] == 'Convergence')}; "
                            f"our gate uses R-hat <= {th['rhat_max']} plus ESS and divergences"),
            "GoodnessOfFit": "Meridian reports R2/MAPE for guidance only (no pass/fail bar); our in-sample and CV gates are explicit",
            "BayesianPPP": "Meridian checks only the total outcome; our gate checks weekly interval coverage"}
  meridian_checks = [{**c, "comparison": looser.get(c["check"], "")} for c in health["checks"]]
  for c in meridian_checks:
    c["details"] = {k: v for k, v in c["details"].items() if k != "correlation_matrix"}

  blocking_fail = [g["gate"] for g in gates if g["blocking"] and g["status"] == "FAIL"]
  result = {"model": name, "thresholds": T,
            "threshold_basis": basis, "gates": gates, "insample": insample,
            "meridian_health": {"overall_status": health["overall_status"],
                                "health_score": health["health_score"],
                                "checks": meridian_checks},
            "blocking_failures": blocking_fail, "passed": not blocking_fail}
  write_json(result, OUT / "gates.json")
  times = [str(t) for t in np.asarray(mmm.input_data.time)]
  pd.DataFrame({"time": times, "observed": obs.sum(0), "mean": m.sum(0), "ci_lo": nlo,
                "ci_hi": nhi}).to_csv(OUT / "insample_national.csv", index=False)
  for g in gates:
    print(f"{g['status']:5s} {'(blocking)' if g['blocking'] else '          '} {g['gate']}")
  print(f"Blocking failures: {blocking_fail or 'none'}")
  sys.exit(1 if blocking_fail else 0)


if __name__ == "__main__":
  main()
