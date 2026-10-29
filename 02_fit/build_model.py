"""Meridian model build script (generated with the meridian-model-building skill).

Builds the single production model defined in config/model.toml. Steps follow the
skill: load data and map columns (DataFrameInputDataBuilder), configure ModelSpec,
sample the prior, fit with sample_posterior and save with meridian_serde.save_meridian.

Usage (run from the project root with PYTHONPATH=. so that `common` is found;
the Makefile does this):
  PYTHONPATH=. .venv/bin/python 02_fit/build_model.py                    # full-data fit
  PYTHONPATH=. .venv/bin/python 02_fit/build_model.py --train-weeks 104  # CV refit on weeks 1-104
  PYTHONPATH=. .venv/bin/python 02_fit/build_model.py --calibrated       # HYPOTHETICAL lift test
"""

import argparse
import json
import os
import time

import pandas as pd
from meridian.model.model import Meridian
from meridian.schema.serde import meridian_serde

from common import RAW_CSV, TIME, build_input_data, model_dir
from model_spec import CONFIG, make_spec


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-weeks", type=int, default=None,
                      help="CV refit: keep only the first N weeks.")
  parser.add_argument("--calibrated", action="store_true",
                      help="Optional: ROI prior calibrated by the lift tests file.")
  args = parser.parse_args()
  sampler = {k: v for k, v in CONFIG["sampler"].items() if k != "n_prior_draws"}

  output_dir = model_dir("calibrated" if args.calibrated else "model", args.train_weeks)
  os.makedirs(output_dir, exist_ok=True)

  # 1. Load data & build InputData (approved column mapping).
  df = pd.read_csv(RAW_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
  if args.train_weeks:
    weeks = sorted(df[TIME].unique())[:args.train_weeks]
    df = df[df[TIME].isin(weeks)]
  input_data = build_input_data(df)

  # 2. ModelSpec + Meridian.
  mmm = Meridian(input_data=input_data, model_spec=make_spec(input_data, args.calibrated))

  # 3. Prior sampling (fixed seed).
  mmm.sample_prior(n_draws=CONFIG["sampler"]["n_prior_draws"], seed=CONFIG["sampler"]["seed"])

  # 4. Fit.
  t0 = time.time()
  mmm.sample_posterior(**sampler)
  runtime = time.time() - t0

  # 5. Save.
  save_path = output_dir / "meridian_model.binpb"
  meridian_serde.save_meridian(mmm, str(save_path))
  with open(output_dir / "fit_info.json", "w") as f:
    json.dump({"model": CONFIG["model"]["name"], "calibrated": args.calibrated,
               "train_weeks": args.train_weeks, "sampler": sampler,
               "n_prior_draws": CONFIG["sampler"]["n_prior_draws"],
               "knot_locations": [int(k) for k in mmm.knot_info.knot_locations],
               "n_knots": int(len(mmm.knot_info.knot_locations)),
               "runtime_seconds": runtime}, f, indent=2)
  print(f"Model saved to {save_path} ({runtime / 60:.1f} min)")


if __name__ == "__main__":
  main()
