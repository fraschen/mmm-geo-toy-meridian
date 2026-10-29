"""Convergence gate for the production model (`make convergence`).

Rules and thresholds: config/gates.toml (gates.convergence); the check itself is
`convergence()` in common.py, also used for the validation refits. Writes
outputs/02_fit/convergence.json and exits non-zero if the model fails.
"""

import sys

from common import OUTPUTS, convergence, load_model, write_json


def main():
  conv = convergence(load_model().inference_data)
  write_json(conv, OUTPUTS / "02_fit" / "convergence.json")
  print(f"converged={conv['converged']} max_rhat={conv['max_rhat']:.4f} "
        f"min_ess_bulk={conv['min_ess_bulk']:.0f} min_ess_tail={conv['min_ess_tail']:.0f} "
        f"divergences={conv['divergences']}")
  sys.exit(0 if conv["converged"] else 1)


if __name__ == "__main__":
  main()
