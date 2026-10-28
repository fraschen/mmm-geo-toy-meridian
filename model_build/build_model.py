"""Meridian model build script (generated with the meridian-model-building skill).

Steps follow the skill: load data and map columns (DataFrameInputDataBuilder),
configure ModelSpec, sample the prior, run Meridian EDA, and (for candidate
specs, Phase 3) fit and save with meridian_serde.

Usage (run from the project root):
  .venv/bin/python model_build/build_model.py --spec eda_default --eda-only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from common import SEED, build_input_data  # noqa: E402
from meridian.model.eda import meridian_eda  # noqa: E402
from meridian.model.model import Meridian  # noqa: E402
from meridian.model.prior_distribution import PriorDistribution  # noqa: E402
from meridian.model.spec import ModelSpec  # noqa: E402


def make_spec(name: str) -> ModelSpec:
  if name == "eda_default":
    # Meridian defaults, used only to drive the EDA report's prior checks.
    # Not a modelling candidate (candidates are defined in Phase 2).
    return ModelSpec(prior=PriorDistribution())
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
  mmm.sample_prior(n_draws=500, seed=SEED)

  # 4. Meridian EDA report.
  eda = meridian_eda.MeridianEDA(mmm)
  eda.generate_and_save_report(filename="eda.html", filepath=output_dir)
  print(f"EDA report written to {os.path.join(output_dir, 'eda.html')}")

  if args.eda_only:
    return
  raise NotImplementedError("Fitting is added in Phase 3 after spec approval.")


if __name__ == "__main__":
  main()
