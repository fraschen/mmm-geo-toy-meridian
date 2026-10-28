"""Convergence gate for the production model (and its CV and calibrated refits).

Rules and thresholds: config/gates.toml (gates.convergence). For every element of
every posterior variable: R-hat, bulk ESS and tail ESS; plus the share of
divergent transitions. Elements constant across all draws (e.g. the baseline
geo's intercept, fixed at 0) are skipped. A fit that fails records only these
diagnostics: its ROI, fit and contributions are never reported.

`python scripts/check_convergence.py` checks the production fit, writes
outputs/model/convergence.json and exits non-zero if it fails.
"""

from __future__ import annotations

import sys

import arviz as az
import numpy as np

from common import OUTPUTS, load_gates_config, load_model, write_json

_CFG = load_gates_config()
_CT = _CFG["gates"]["convergence"]["thresholds"]
RHAT_MAX, ESS_MIN = _CT["rhat_max"], _CT["ess_min"]
DIVERGENT_SHARE_MAX = _CT["divergent_share_max"]
GATE_NOTE = (f"R-hat <= {RHAT_MAX}, bulk/tail ESS >= {ESS_MIN}, <= {DIVERGENT_SHARE_MAX:.1%} "
             "divergent draws (config/gates.toml). " + _CFG["meta"]["note"])


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
                         "note": GATE_NOTE},
          "n_chains": int(post.sizes["chain"]), "n_draws_total": n_draws,
          **checks, "n_failing_elements": n_fail, "passed": passed,
          "converged": all(passed.values()),
          "skipped_constant_variables": constant, "per_variable": per_var}



def main():
  conv = convergence(load_model().inference_data)
  write_json(conv, OUTPUTS / "model" / "convergence.json")
  print(f"converged={conv['converged']} max_rhat={conv['max_rhat']:.4f} "
        f"min_ess_bulk={conv['min_ess_bulk']:.0f} min_ess_tail={conv['min_ess_tail']:.0f} "
        f"divergences={conv['divergences']}")
  sys.exit(0 if conv["converged"] else 1)


if __name__ == "__main__":
  main()
