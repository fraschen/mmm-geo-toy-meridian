"""Meridian model build script (generated with the meridian-model-building skill).

Builds the single production model defined in config/model.toml. Steps follow the
skill: load data and map columns (DataFrameInputDataBuilder), configure ModelSpec,
sample the prior, fit with sample_posterior and save with meridian_serde.save_meridian.

Usage (run from the project root):
  .venv/bin/python model_build/build_model.py                   # full-data fit
  .venv/bin/python model_build/build_model.py --train-weeks 104 # CV refit on weeks 1-104
  .venv/bin/python model_build/build_model.py --calibrated      # optional: HYPOTHETICAL lift test
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd  # noqa: E402
from common import (RAW_CSV, ROOT, TIME, build_input_data, load_model_config,  # noqa: E402
                    model_dir)
from meridian import backend  # noqa: E402
from meridian.model.model import Meridian  # noqa: E402
from meridian.model.prior_distribution import PriorDistribution  # noqa: E402
from meridian.model.spec import ModelSpec  # noqa: E402
from meridian.schema.serde import meridian_serde  # noqa: E402

CONFIG = load_model_config()


def _dist(name, cfg):
  """A tfd distribution from a config entry such as {dist = "LogNormal", loc = 0.2, scale = 0.9}."""
  params = {k: np.float64(v) for k, v in cfg.items() if k != "dist"}
  return getattr(backend.tfd, cfg["dist"])(**params, name=name)


def calibrated_roi_prior(input_data, max_lag):
  """ROI prior from lift tests via Meridian's CalibrationBuilder.

  Uncalibrated channels keep the configured ROI prior; the same distribution is
  the baseline prior that each experiment updates.
  """
  from meridian.model.calibration import prior_builder
  tfd = backend.tfd
  base = CONFIG["model"]["priors"]["roi_m"]
  default = _dist("roi_m", base)
  builder = prior_builder.CalibrationBuilder(
      input_data, custom_prior=default, baseline_prior=default,
      adstock_decay_spec=CONFIG["model"]["spec"]["adstock_decay_spec"], max_lag=max_lag)
  tests = pd.read_csv(ROOT / CONFIG["calibration"]["lift_tests"])
  for channel, g in tests.groupby("channel"):
    builder = builder.with_incrementality_experiment_result(
        channel, point_estimates=g["point_estimate_roi"].tolist(),
        standard_errors=g["standard_error"].tolist(),
        experiment_kpi_types=g["kpi_type"].tolist(),
        experiment_total_spends=g["total_spend"].tolist(),
        experiment_start_dates=g["start_date"].tolist(),
        experiment_end_dates=g["end_date"].tolist())
  calibrated = builder.build().roi_m
  # google-meridian 2.1.0: the mixed-family CalibratedDistribution fails under JAX
  # JIT in sample_posterior (jnp.split on a traced value). Use the builder's fitted
  # per-channel prior, moment-matched to a LogNormal, in a jit-safe LogNormal vector.
  channels = [str(c) for c in np.asarray(input_data.media_channel)]
  loc, scale = np.full(len(channels), base["loc"]), np.full(len(channels), base["scale"])
  for channel in tests["channel"].unique():
    i = channels.index(channel)
    comp = calibrated._distributions[i]
    m = float(np.asarray(comp.mean()).ravel()[0])
    sd = float(np.asarray(comp.stddev()).ravel()[0])
    s2 = np.log(1 + sd ** 2 / m ** 2)
    loc[i], scale[i] = np.log(m) - s2 / 2, np.sqrt(s2)
  return tfd.LogNormal(loc.astype(np.float64), scale.astype(np.float64), name="roi_m")


def make_spec(input_data=None, calibrated: bool = False) -> ModelSpec:
  """The production ModelSpec from config/model.toml (optionally lift-test calibrated)."""
  spec = dict(CONFIG["model"]["spec"])
  priors = {name: _dist(name, cfg) for name, cfg in CONFIG["model"]["priors"].items()}
  if calibrated:
    priors["roi_m"] = calibrated_roi_prior(input_data, max_lag=spec["max_lag"])
  return ModelSpec(prior=PriorDistribution(**priors), **spec)


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
