"""Budget optimization scenarios (meridian-budget-optimization skill).

Fixed total budget = last 52 weeks' paid spend; main scenario +/-20% per channel,
stress scenario +/-50%. Writes Meridian's optimization HTML reports and, per
scenario, a JSON with the allocation, the expected revenue gain with its 94%
credible interval and P(gain > 0), and a budget-conservation check.

The gain distribution is recomputed per posterior draw the way the optimizer
evaluates a spend multiplier (historical media units scaled by optimal /
historical spend in every period; incremental revenue summed over the 52-week
window; paid channels only), for the optimal and the historical allocation.

Usage (from the project root):
  .venv/bin/python scripts/04_scenarios/run_budget_optimization.py
"""

import json
import os

import numpy as np
from meridian import backend
from meridian.analysis import optimizer
from meridian.analysis import tensors
from meridian.analysis.analyzer import Analyzer
from meridian.schema.serde import meridian_serde

MODEL_PATH = "models/model/meridian_model.binpb"  # production model (config/model.toml)
REPORT_DIR = "reports/meridian"
OUT_DIR = "outputs/04_scenarios"
START_DATE, END_DATE = "2023-01-23", "2024-01-15"
SCENARIOS = {"main": 0.2, "stress": 0.5}
CI = (0.03, 0.97)


def gain_draws(model, ratio, times):
  an = Analyzer(model_context=model.model_context, inference_data=model.inference_data)
  data = model.input_data
  media = np.asarray(data.media)
  rpk = backend.to_tensor(np.asarray(data.revenue_per_kpi), dtype=backend.float_dtype)
  time = backend.to_tensor(np.asarray(data.time))

  def total(r):
    new = tensors.DataTensors(
        media=backend.to_tensor(media * np.asarray(r)[None, None, :], dtype=backend.float_dtype),
        revenue_per_kpi=rpk, time=time)
    inc = np.asarray(an.incremental_outcome(new_data=new, selected_times=times, use_kpi=False,
                                            include_non_paid_channels=False))
    return inc  # (chain, draw, channel)

  opt, hist = total(ratio), total(np.ones_like(ratio))
  return opt, hist


def main():
  os.makedirs(REPORT_DIR, exist_ok=True)
  os.makedirs(OUT_DIR, exist_ok=True)

  # 1. Load the model.
  model = meridian_serde.load_meridian(MODEL_PATH)
  times = [t for t in np.asarray(model.input_data.time) if START_DATE <= str(t) <= END_DATE]
  spend = np.asarray(model.input_data.media_spend)  # (geo, time, channel)
  t_idx = [i for i, t in enumerate(np.asarray(model.input_data.time)) if str(t) in times]
  budget = float(spend[:, t_idx, :].sum())
  channels = [str(c) for c in np.asarray(model.input_data.media_channel)]

  budget_optimizer = optimizer.BudgetOptimizer(model)
  for name, bound in SCENARIOS.items():
    # 2. Fixed-budget optimization.
    res = budget_optimizer.optimize(
        use_posterior=True, fixed_budget=True, budget=budget,
        start_date=START_DATE, end_date=END_DATE,
        spend_constraint_lower=bound, spend_constraint_upper=bound, use_kpi=False)
    res.output_optimization_summary(filename=f"optimization_{name}.html", filepath=REPORT_DIR)

    hist_spend = np.asarray(res.nonoptimized_data["spend"])
    opt_spend = np.asarray(res.optimized_data["spend"])
    ratio = opt_spend / hist_spend
    opt, hist = gain_draws(model, ratio, times)
    gain = (opt.sum(-1) - hist.sum(-1)).ravel()
    opt_mean = opt.sum(-1).mean()
    reported = float(np.asarray(res.optimized_data["incremental_outcome"].sel(metric="mean")).sum())
    out = {
        "scenario": name, "spend_constraint": bound, "period": [START_DATE, END_DATE],
        "n_weeks": len(times), "budget": budget,
        "budget_conserved": {"optimal_total": float(opt_spend.sum()),
                             "historical_total": float(hist_spend.sum()),
                             "relative_difference": float(opt_spend.sum() / budget - 1)},
        "channels": {c: {"historical_spend": float(hist_spend[i]),
                         "optimal_spend": float(opt_spend[i]),
                         "change_pct": float(ratio[i] - 1),
                         "lower_bound": float(hist_spend[i] * (1 - bound)),
                         "upper_bound": float(hist_spend[i] * (1 + bound)),
                         "at_bound": bool(abs(abs(ratio[i] - 1) - bound) < 1e-3),
                         "incremental_revenue_historical": float(hist[..., i].mean()),
                         "incremental_revenue_optimal": float(opt[..., i].mean())}
                     for i, c in enumerate(channels)},
        "incremental_revenue_historical_mean": float(hist.sum(-1).mean()),
        "incremental_revenue_optimal_mean": float(opt_mean),
        "gain": {"mean": float(gain.mean()), "ci_lo": float(np.quantile(gain, CI[0])),
                 "ci_hi": float(np.quantile(gain, CI[1])),
                 "p_gain_gt_0": float((gain > 0).mean()),
                 "pct_of_historical_incremental": float(gain.mean() / hist.sum(-1).mean())},
        "check_optimizer_reported_optimal_mean": reported,
        "check_relative_difference": float(opt_mean / reported - 1),
        "report": f"{REPORT_DIR}/optimization_{name}.html",
    }
    with open(os.path.join(OUT_DIR, f"scenario_{name}.json"), "w") as f:
      json.dump(out, f, indent=2)
    print(f"{name}: gain {out['gain']['mean']:,.0f} [{out['gain']['ci_lo']:,.0f}, "
          f"{out['gain']['ci_hi']:,.0f}], P(gain>0)={out['gain']['p_gain_gt_0']:.2f}; "
          f"budget diff {out['budget_conserved']['relative_difference']:.2e}; "
          f"check vs optimizer {out['check_relative_difference']:.2e}")


if __name__ == "__main__":
  main()
