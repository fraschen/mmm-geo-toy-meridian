"""The production ModelSpec, built from config/model.toml.

Used by scripts/02_fit/build_model.py (fitting) and docker/check_sampling.py. Optionally the
ROI prior is calibrated with the HYPOTHETICAL lift test (data/lift_tests/).
"""

import numpy as np
import pandas as pd
from meridian import backend
from meridian.model.prior_distribution import PriorDistribution
from meridian.model.spec import ModelSpec

from common import ROOT, load_model_config

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
