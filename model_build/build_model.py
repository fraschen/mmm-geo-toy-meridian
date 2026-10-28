"""Meridian model build script (generated with the meridian-model-building skill).

Steps follow the skill: load data and map columns (DataFrameInputDataBuilder),
configure ModelSpec, sample the prior, run Meridian EDA, fit with
sample_posterior and save with meridian_serde.save_meridian.

Specs (approved at the Phase 2 checkpoint, see reports/candidates.md):
  eda_default          Meridian defaults; prior draws + EDA report only.
  aks_default_pool     C1: AKS knots, eta_m ~ HalfNormal(1) (default).
  linear_default_pool  C2: knots=2 (linear trend), eta_m ~ HalfNormal(1).
  aks_strong_pool      C3: AKS knots, eta_m ~ HalfNormal(0.1).

Usage (run from the project root):
  .venv/bin/python model_build/build_model.py --spec eda_default --eda-only
  .venv/bin/python model_build/build_model.py --spec aks_default_pool
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from common import SEED, build_input_data  # noqa: E402
from meridian import backend  # noqa: E402
from meridian.model.eda import meridian_eda  # noqa: E402
from meridian.model.model import Meridian  # noqa: E402
from meridian.model.prior_distribution import PriorDistribution  # noqa: E402
from meridian.model.spec import ModelSpec  # noqa: E402
from meridian.schema.serde import meridian_serde  # noqa: E402

# Approved MCMC settings (identical for every candidate).
SAMPLER = dict(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=SEED)
N_PRIOR_DRAWS = 500

CANDIDATES = ["aks_default_pool", "linear_default_pool", "aks_strong_pool"]


def make_spec(name: str, **overrides) -> ModelSpec:
  tfd = backend.tfd
  if name == "eda_default":
    # Meridian defaults, used only to drive the EDA report's prior checks.
    return ModelSpec(prior=PriorDistribution())
  common = dict(media_effects_dist="log_normal", max_lag=8,
                adstock_decay_spec="geometric", hill_before_adstock=False,
                media_prior_type="roi", unique_sigma_for_each_geo=False)
  common.update(overrides)
  if name == "aks_default_pool":
    return ModelSpec(prior=PriorDistribution(), enable_aks=True, **common)
  if name == "linear_default_pool":
    return ModelSpec(prior=PriorDistribution(), knots=2, **common)
  if name == "aks_strong_pool":
    return ModelSpec(prior=PriorDistribution(
        eta_m=tfd.HalfNormal(scale=np.float64(0.1), name="eta_m")),
                     enable_aks=True, **common)
  raise ValueError(f"Unknown spec {name!r}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--spec", default="eda_default")
  parser.add_argument("--eda-only", action="store_true")
  args = parser.parse_args()

  output_dir = os.path.join("model_build", args.spec)
  os.makedirs(output_dir, exist_ok=True)

  # 1. Load data & build InputData (approved column mapping).
  input_data = build_input_data()

  # 2. ModelSpec + Meridian.
  mmm = Meridian(input_data=input_data, model_spec=make_spec(args.spec))

  # 3. Prior sampling (fixed seed).
  mmm.sample_prior(n_draws=N_PRIOR_DRAWS, seed=SEED)

  # 4. Meridian EDA report.
  eda = meridian_eda.MeridianEDA(mmm)
  eda.generate_and_save_report(filename="eda.html", filepath=output_dir)
  print(f"EDA report written to {os.path.join(output_dir, 'eda.html')}")
  if args.eda_only:
    return

  # 5. Fit.
  t0 = time.time()
  mmm.sample_posterior(**SAMPLER)
  runtime = time.time() - t0

  # 6. Save.
  save_path = os.path.join(output_dir, "meridian_model.binpb")
  meridian_serde.save_meridian(mmm, save_path)
  with open(os.path.join(output_dir, "fit_info.json"), "w") as f:
    json.dump({"spec": args.spec, "sampler": SAMPLER, "n_prior_draws": N_PRIOR_DRAWS,
               "knot_locations": [int(k) for k in mmm.knot_info.knot_locations],
               "n_knots": int(len(mmm.knot_info.knot_locations)),
               "runtime_seconds": runtime}, f, indent=2)
  print(f"Model saved to {save_path} ({runtime / 60:.1f} min)")


if __name__ == "__main__":
  main()
