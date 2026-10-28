# HISTORICAL RECORD (not run by the pipeline). This is the script that compared the
# candidate specifications and selected R1; its outputs are in this folder
# (selection.json, convergence_*.json, loo_pointwise_*.csv). The candidate models
# are not kept: the production pipeline fits only config/model.toml.
"""Phase 3: convergence gate and PSIS-LOO selection (protocol in reports/candidates.md).

For each candidate, in the fixed complexity order:
  1. Convergence gate (config/gates.toml, gates.convergence) for every element of
     every posterior variable (elements constant across all draws are skipped).
     A failing candidate records only these diagnostics.
  2. Converged candidates: pointwise log-likelihood Normal(y_gt | mu_gt, sigma) on
     Meridian's transformed KPI (kpi_scaled, the likelihood node "y"), PSIS-LOO with
     ArviZ; pointwise values and Pareto k saved.
  3. Start from the simplest converged candidate; move to the next more complex one
     only if its ELPD exceeds the current choice's by more than se_rule SE of the paired
     difference (config/gates.toml, selection).
"""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

from common import MODEL_BUILD, OUTPUTS, load_gates_config, write_json

# Rounds, in the order they were run (all 4 chains x 1000 kept draws, seed 20260128).
ROUNDS = {
    "round_1": ["linear_default_pool", "aks_strong_pool", "aks_default_pool"],
    "round_2": ["linear_strong_pool_lag4", "linear_strong_pool"],
}
ROUND_NOTE = {
    "round_1": "C1-C3: knots (AKS vs linear) x media pooling (default vs strong); default sampler.",
    "round_2": ("R1-R2: after round 1 failed in the baseline parameters, linear knots and strong "
                "pooling (the docs' convergence checklist), with max lag 4 vs 8; default sampler."),
}
# Complexity order for selection, simplest first.
ORDER = ["linear_strong_pool_lag4", "linear_strong_pool", "linear_default_pool",
         "aks_strong_pool", "aks_default_pool"]
LABEL = {"linear_strong_pool_lag4": "R1", "linear_strong_pool": "R2",
         "linear_default_pool": "C2", "aks_strong_pool": "C3", "aks_default_pool": "C1"}
# Convergence gate and selection rule: config/gates.toml.
_CFG = load_gates_config()
_CT = _CFG["gates"]["convergence"]["thresholds"]
RHAT_MAX, ESS_MIN = _CT["rhat_max"], _CT["ess_min"]
DIVERGENT_SHARE_MAX = _CT["divergent_share_max"]
SE_RULE = _CFG["selection"]["thresholds"]["se_rule"]
GATE_NOTE = (f"R-hat <= {RHAT_MAX}, bulk/tail ESS >= {ESS_MIN}, <= {DIVERGENT_SHARE_MAX:.1%} "
             "divergent draws (config/gates.toml). " + _CFG["meta"]["note"])
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
            "divergences": div <= DIVERGENT_SHARE_MAX * n_draws}
  return {"thresholds": {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN,
                         "divergences_max": DIVERGENT_SHARE_MAX * n_draws,
                         "preregistered": {"rhat_max": 1.01, "ess_min": 400, "divergences": 0},
                         "note": GATE_NOTE},
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
  for name in ORDER:
    path = MODEL_BUILD / name / "meridian_model.binpb"
    if not path.exists():
      # Round-1 models are not refit by `make all` (`make fit-round1` does); their
      # committed convergence diagnostics are used. None of them converged.
      conv[name] = json.load(open(OUT / f"convergence_{name}.json"))
      print(f"{LABEL[name]} {name}: converged={conv[name]['converged']} (stored diagnostics)")
      continue
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
    basis = ("convergence: the only converged candidate (won by convergence, not by fit)"
             if len(converged) == 1 else "PSIS-LOO with the 2-SE simplicity rule")
  else:
    basis = "none: no candidate converged"
  summary = lambda names: {n: {k: conv[n][k] for k in ("max_rhat", "min_ess_bulk",
                                                       "min_ess_tail", "divergences", "passed")}
                           for n in names}
  result = {"labels": LABEL, "gate_note": GATE_NOTE, "rounds": ROUNDS, "round_notes": ROUND_NOTE,
            "order_simplest_first": ORDER,
            "converged": {n: conv[n]["converged"] for n in ORDER},
            "convergence_summary": summary(ORDER), "loo": loo, "comparison_steps": steps,
            "selected": selected, "selection_basis": basis}
  write_json(result, OUT / "selection.json")
  print(json.dumps({"selected": selected, "basis": basis, "steps": steps}, indent=2))


if __name__ == "__main__":
  main()
