"""Meridian model build script (generated with the meridian-model-building skill).

Steps follow the skill: load data and map columns (DataFrameInputDataBuilder),
configure ModelSpec, sample the prior, run Meridian EDA, fit with
sample_posterior and save with meridian_serde.save_meridian.

Specs (approved at the Phase 2 checkpoint, see reports/candidates.md):
  eda_default          Meridian defaults; prior draws + EDA report only.
  aks_default_pool     C1: AKS knots, eta_m ~ HalfNormal(1) (default).
  linear_default_pool  C2: knots=2 (linear trend), eta_m ~ HalfNormal(1).
  aks_strong_pool      C3: AKS knots, eta_m ~ HalfNormal(0.1).
  Round 2 (after C1-C3 all failed the convergence gate):
  linear_strong_pool_lag4  R1: knots=2, eta_m ~ HalfNormal(0.1), max_lag=4.
  linear_strong_pool       R2: knots=2, eta_m ~ HalfNormal(0.1), max_lag=8.
  Phase 8:
  linear_strong_pool_lag4_calibrated  R1 with ROI priors calibrated by
      CalibrationBuilder from data/lift_tests/lift_tests.csv (HYPOTHETICAL test).

Usage (run from the project root):
  .venv/bin/python model_build/build_model.py --spec eda_default --eda-only
  .venv/bin/python model_build/build_model.py --spec aks_default_pool
  # Phase 4 CV fold: fit on the first N weeks only (all scaling from those weeks).
  .venv/bin/python model_build/build_model.py --spec linear_strong_pool_lag4 --train-weeks 104
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd  # noqa: E402
from common import RAW_CSV, SEED, TIME, build_input_data  # noqa: E402
from meridian import backend  # noqa: E402
from meridian.model.eda import meridian_eda  # noqa: E402
from meridian.model.model import Meridian  # noqa: E402
from meridian.model.prior_distribution import PriorDistribution  # noqa: E402
from meridian.model.spec import ModelSpec  # noqa: E402
from meridian.schema.serde import meridian_serde  # noqa: E402

# Approved MCMC settings. "default" was used for rounds 1-2 and the first CV run;
# "long" (warm-up of the Meridian getting-started notebook) was approved post hoc
# for the round-3 refit of C1-C3.
SAMPLERS = {
    "default": dict(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=SEED),
    "long": dict(n_chains=4, n_adapt=1000, n_burnin=500, n_keep=1000, seed=SEED),
}
N_PRIOR_DRAWS = 500

CANDIDATES = ["aks_default_pool", "linear_default_pool", "aks_strong_pool",
              "linear_strong_pool_lag4", "linear_strong_pool"]


LIFT_TESTS = os.path.join(os.path.dirname(__file__), "..", "data", "lift_tests",
                          "lift_tests.csv")


def calibrated_roi_prior(input_data, max_lag):
  """ROI prior from lift tests via Meridian's CalibrationBuilder.

  Uncalibrated channels keep the default LogNormal(0.2, 0.9); the same
  distribution is the baseline prior that each experiment updates.
  """
  from meridian.model.calibration import prior_builder
  tfd = backend.tfd
  default = tfd.LogNormal(np.float64(0.2), np.float64(0.9), name="roi_m")
  builder = prior_builder.CalibrationBuilder(
      input_data, custom_prior=default, baseline_prior=default,
      adstock_decay_spec="geometric", max_lag=max_lag)
  tests = pd.read_csv(LIFT_TESTS)
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
  loc, scale = np.full(len(channels), 0.2), np.full(len(channels), 0.9)
  for channel in tests["channel"].unique():
    i = channels.index(channel)
    comp = calibrated._distributions[i]
    m = float(np.asarray(comp.mean()).ravel()[0])
    sd = float(np.asarray(comp.stddev()).ravel()[0])
    s2 = np.log(1 + sd ** 2 / m ** 2)
    loc[i], scale[i] = np.log(m) - s2 / 2, np.sqrt(s2)
  return tfd.LogNormal(loc.astype(np.float64), scale.astype(np.float64), name="roi_m")


def make_spec(name: str, input_data=None, **overrides) -> ModelSpec:
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
  strong_pool = PriorDistribution(
      eta_m=tfd.HalfNormal(scale=np.float64(0.1), name="eta_m"))
  if name == "linear_strong_pool_lag4":
    return ModelSpec(prior=strong_pool, knots=2, **{**common, "max_lag": 4})
  if name == "linear_strong_pool":
    return ModelSpec(prior=strong_pool, knots=2, **common)
  if name == "linear_strong_pool_lag4_calibrated":
    prior = PriorDistribution(
        eta_m=tfd.HalfNormal(scale=np.float64(0.1), name="eta_m"),
        roi_m=calibrated_roi_prior(input_data, max_lag=4))
    return ModelSpec(prior=prior, knots=2, **{**common, "max_lag": 4})
  raise ValueError(f"Unknown spec {name!r}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--spec", default="eda_default")
  parser.add_argument("--eda-only", action="store_true")
  parser.add_argument("--sampler", choices=list(SAMPLERS), default="default")
  parser.add_argument("--train-weeks", type=int, default=None,
                      help="CV fold: keep only the first N weeks; skips the EDA report.")
  args = parser.parse_args()

  sampler = SAMPLERS[args.sampler]
  run_name = args.spec if args.sampler == "default" else f"{args.spec}__{args.sampler}"
  if args.train_weeks:
    output_dir = os.path.join("model_build", "cv", f"{run_name}_train{args.train_weeks}")
  else:
    output_dir = os.path.join("model_build", run_name)
  os.makedirs(output_dir, exist_ok=True)

  # 1. Load data & build InputData (approved column mapping).
  df = pd.read_csv(RAW_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
  if args.train_weeks:
    weeks = sorted(df[TIME].unique())[:args.train_weeks]
    df = df[df[TIME].isin(weeks)]
  input_data = build_input_data(df)

  # 2. ModelSpec + Meridian.
  mmm = Meridian(input_data=input_data, model_spec=make_spec(args.spec, input_data))

  # 3. Prior sampling (fixed seed).
  mmm.sample_prior(n_draws=N_PRIOR_DRAWS, seed=SEED)

  # 4. Meridian EDA report (full-data fits only).
  if not args.train_weeks:
    eda = meridian_eda.MeridianEDA(mmm)
    eda.generate_and_save_report(filename="eda.html", filepath=output_dir)
    print(f"EDA report written to {os.path.join(output_dir, 'eda.html')}")
  if args.eda_only:
    return

  # 5. Fit.
  t0 = time.time()
  mmm.sample_posterior(**sampler)
  runtime = time.time() - t0

  # 6. Save.
  save_path = os.path.join(output_dir, "meridian_model.binpb")
  meridian_serde.save_meridian(mmm, save_path)
  with open(os.path.join(output_dir, "fit_info.json"), "w") as f:
    json.dump({"spec": args.spec, "train_weeks": args.train_weeks, "sampler_name": args.sampler, "sampler": sampler, "n_prior_draws": N_PRIOR_DRAWS,
               "knot_locations": [int(k) for k in mmm.knot_info.knot_locations],
               "n_knots": int(len(mmm.knot_info.knot_locations)),
               "runtime_seconds": runtime}, f, indent=2)
  print(f"Model saved to {save_path} ({runtime / 60:.1f} min)")


if __name__ == "__main__":
  main()
