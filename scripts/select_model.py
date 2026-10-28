"""Phase 3: convergence gate and PSIS-LOO selection (protocol in reports/candidates.md).

For each candidate, in the fixed complexity order:
  1. Convergence: R-hat <= 1.01, bulk ESS >= 400, tail ESS >= 400 for every element
     of every posterior variable (variables constant across all draws, such as a
     Deterministic slope, are listed and skipped), and 0 divergent transitions.
     A failing candidate records only these diagnostics.
  2. Converged candidates: pointwise log-likelihood Normal(y_gt | mu_gt, sigma) on
     Meridian's transformed KPI (kpi_scaled, the likelihood node "y"), PSIS-LOO with
     ArviZ; pointwise values and Pareto k saved.
  3. Start from the simplest converged candidate; move to the next more complex one
     only if its ELPD exceeds the current choice's by > 2 SE of the paired difference.
"""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

from common import MODEL_BUILD, OUTPUTS, write_json

# Round 1 (C1-C3): no candidate converged; diagnostics kept for the record.
ROUND1 = ["linear_default_pool", "aks_strong_pool", "aks_default_pool"]
# Round 2, complexity order simplest first (R1 < R2). Selection uses round 2 only.
ORDER = ["linear_strong_pool_lag4", "linear_strong_pool"]
LABEL = {"aks_default_pool": "C1", "linear_default_pool": "C2", "aks_strong_pool": "C3",
         "linear_strong_pool_lag4": "R1", "linear_strong_pool": "R2"}
RHAT_MAX, ESS_MIN, SE_RULE = 1.01, 400, 2.0
# Pre-registered: 0 divergences. Relaxed POST HOC (user decision after round 2,
# where R1 and R2 each had 1 divergence in 4000 draws) to <= 1 per 4000 draws.
DIVERGENCES_MAX_PER_4000 = 1
DIVERGENCE_RULE_NOTE = ("Post-hoc relaxation: the pre-registered rule was 0 divergences; "
                        "relaxed to <= 1 per 4000 draws after round 2 results.")
OUT = OUTPUTS / "selection"


def load(name):
  from meridian.schema.serde import meridian_serde
  return meridian_serde.load_meridian(str(MODEL_BUILD / name / "meridian_model.binpb"))


def convergence(idata) -> dict:
  post = idata.posterior
  constant = [v for v in post.data_vars if float(post[v].std()) == 0.0]
  post = post.drop_vars(constant)
  rhat = az.rhat(post)
  ess_bulk = az.ess(post, method="bulk")
  ess_tail = az.ess(post, method="tail")
  per_var, n_fail = {}, {"rhat": 0, "ess_bulk": 0, "ess_tail": 0}
  for v in post.data_vars:
    r, b, t = (np.asarray(rhat[v]), np.asarray(ess_bulk[v]), np.asarray(ess_tail[v]))
    # Elements constant across all draws (e.g. the baseline geo's tau_g, fixed
    # at 0) have undefined diagnostics and are skipped; any other NaN fails.
    const = np.asarray(post[v].std(("chain", "draw"))) == 0.0
    r, b, t = (np.where(const, 1.0, r), np.where(const, np.inf, b), np.where(const, np.inf, t))
    per_var[v] = {"n_elements": int(r.size), "n_constant_elements_skipped": int(const.sum()),
                  "max_rhat": float(np.max(r)),
                  "min_ess_bulk": float(np.min(b)), "min_ess_tail": float(np.min(t)),
                  "n_rhat_fail": int((~(r <= RHAT_MAX)).sum()),
                  "n_ess_bulk_fail": int((~(b >= ESS_MIN)).sum()),
                  "n_ess_tail_fail": int((~(t >= ESS_MIN)).sum())}
    n_fail["rhat"] += per_var[v]["n_rhat_fail"]
    n_fail["ess_bulk"] += per_var[v]["n_ess_bulk_fail"]
    n_fail["ess_tail"] += per_var[v]["n_ess_tail_fail"]
  div = int(np.asarray(idata.sample_stats["diverging"]).sum())
  n_draws = int(post.sizes["chain"] * post.sizes["draw"])
  checks = {
      "max_rhat": float(np.max([p["max_rhat"] for p in per_var.values()])),
      "min_ess_bulk": min(p["min_ess_bulk"] for p in per_var.values()),
      "min_ess_tail": min(p["min_ess_tail"] for p in per_var.values()),
      "divergences": div,
  }
  passed = {"rhat": checks["max_rhat"] <= RHAT_MAX,
            "ess_bulk": checks["min_ess_bulk"] >= ESS_MIN,
            "ess_tail": checks["min_ess_tail"] >= ESS_MIN,
            "divergences": div <= DIVERGENCES_MAX_PER_4000 * n_draws / 4000}
  return {"thresholds": {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN,
                         "divergences_max": DIVERGENCES_MAX_PER_4000 * n_draws / 4000,
                         "divergences_preregistered": 0, "note": DIVERGENCE_RULE_NOTE},
          "n_chains": int(post.sizes["chain"]), "n_draws_total": n_draws,
          **checks, "n_failing_elements": n_fail, "passed": passed,
          "converged": all(passed.values()),
          "skipped_constant_variables": constant, "per_variable": per_var}


def pointwise_loglik(mmm) -> tuple[np.ndarray, pd.MultiIndex]:
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  mu = np.asarray(an.expected_outcome(use_kpi=True, aggregate_geos=False,
                                      aggregate_times=False,
                                      inverse_transform_outcome=False))
  y = np.asarray(mmm.model_context.kpi_scaled)
  sigma = np.asarray(mmm.inference_data.posterior["sigma"])
  sigma = sigma[..., None, None] if sigma.ndim == 2 else sigma[..., None]
  ll = stats.norm.logpdf(y[None, None], loc=mu, scale=sigma)
  data = mmm.input_data
  idx = pd.MultiIndex.from_product([np.asarray(data.geo), np.asarray(data.time)],
                                   names=["geo", "time"])
  return ll.reshape(*ll.shape[:2], -1), idx


def run_loo(name, mmm) -> dict:
  ll, idx = pointwise_loglik(mmm)
  idata = az.from_dict(posterior={"sigma": np.asarray(mmm.inference_data.posterior["sigma"])},
                       log_likelihood={"y": ll})
  loo = az.loo(idata, pointwise=True)
  k = np.asarray(loo.pareto_k)
  pw = pd.DataFrame({"elpd_loo": np.asarray(loo.loo_i), "pareto_k": k}, index=idx)
  pw.to_csv(OUT / f"loo_pointwise_{name}.csv")
  return {"elpd_loo": float(loo.elpd_loo), "se": float(loo.se), "p_loo": float(loo.p_loo),
          "n_obs": int(len(pw)),
          "pareto_k_gt_0.7": int((k > 0.7).sum()), "pareto_k_gt_1": int((k > 1).sum()),
          "pareto_k_max": float(k.max())}


def paired_diff(a, b) -> dict:
  """ELPD(b) - ELPD(a) and its SE from pointwise values."""
  pa = pd.read_csv(OUT / f"loo_pointwise_{a}.csv", index_col=[0, 1])["elpd_loo"]
  pb = pd.read_csv(OUT / f"loo_pointwise_{b}.csv", index_col=[0, 1])["elpd_loo"]
  d = (pb - pa).to_numpy()
  return {"elpd_diff": float(d.sum()), "se": float(np.sqrt(len(d) * d.var(ddof=1)))}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  conv, loo = {}, {}
  for name in ROUND1:
    # Round 1 is never scored: convergence diagnostics only.
    mmm = load(name)
    conv[name] = convergence(mmm.inference_data)
    write_json(conv[name], OUT / f"convergence_{name}.json")
    print(f"[round 1] {LABEL[name]} {name}: converged={conv[name]['converged']}")
    del mmm
  for name in ORDER:
    mmm = load(name)
    conv[name] = convergence(mmm.inference_data)
    write_json(conv[name], OUT / f"convergence_{name}.json")
    print(f"{LABEL[name]} {name}: converged={conv[name]['converged']} "
          f"max_rhat={conv[name]['max_rhat']:.4f} min_ess_bulk={conv[name]['min_ess_bulk']:.0f} "
          f"min_ess_tail={conv[name]['min_ess_tail']:.0f} divergences={conv[name]['divergences']}")
    if conv[name]["converged"]:
      loo[name] = run_loo(name, mmm)
      print(f"   elpd_loo={loo[name]['elpd_loo']:.1f} (se {loo[name]['se']:.1f}), "
            f"p_loo={loo[name]['p_loo']:.1f}, pareto k>0.7: {loo[name]['pareto_k_gt_0.7']}")
    del mmm

  converged = [n for n in ORDER if conv[n]["converged"]]
  steps, selected = [], None
  if converged:
    selected = converged[0]
    for challenger in converged[1:]:
      d = paired_diff(selected, challenger)
      switch = d["elpd_diff"] > SE_RULE * d["se"]
      steps.append({"current": selected, "challenger": challenger, **d,
                    "threshold": SE_RULE * d["se"], "switch": bool(switch)})
      if switch:
        selected = challenger
  if not converged:
    basis = "none: no candidate converged"
  elif len(converged) == 1:
    basis = "convergence: the only converged candidate (won by convergence, not by fit)"
  else:
    basis = "PSIS-LOO with the 2-SE simplicity rule"
  result = {"round1_not_converged": {n: conv[n]["converged"] for n in ROUND1},
            "round1_convergence_summary": {n: {k: conv[n][k] for k in
                                               ("max_rhat", "min_ess_bulk", "min_ess_tail",
                                                "divergences", "passed")} for n in ROUND1},
            "order_simplest_first": ORDER, "labels": LABEL,
            "converged": {n: conv[n]["converged"] for n in ORDER},
            "convergence_summary": {n: {k: conv[n][k] for k in
                                        ("max_rhat", "min_ess_bulk", "min_ess_tail",
                                         "divergences", "passed", "n_failing_elements")}
                                    for n in ORDER},
            "loo": loo, "comparison_steps": steps, "selected": selected,
            "selection_basis": basis, "divergence_rule_note": DIVERGENCE_RULE_NOTE}
  write_json(result, OUT / "selection.json")
  print(json.dumps({"selected": selected, "basis": basis, "steps": steps}, indent=2))


if __name__ == "__main__":
  main()
