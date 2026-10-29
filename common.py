"""Shared paths, column mapping, palette and data loading for the study.

The column mapping was approved at the meridian-model-building column-mapping
checkpoint (kpi=conversions, non_revenue + revenue_per_conversion, Promo as a
non-media treatment, Organic_channel0_impression as organic media).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "geo_all_channels.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"

# Where data/raw/geo_all_channels.csv comes from (the last upstream commit touching the file).
# The file is committed; this is provenance only, and nothing is downloaded.
DATA_COMMIT = "4229a279581b11ee09ad6f616324f90778480a25"
DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/"
    f"{DATA_COMMIT}/meridian/data/simulated_data/csv/geo_all_channels.csv"
)
DATA_SHA256 = "d9ee016f7cd21f5c90b50da10a794af91a372b69f881d3caa266edd41fdafbf6"

# Column mapping.
GEO, TIME = "geo", "time"
KPI = "conversions"
REVENUE_PER_KPI = "revenue_per_conversion"
POPULATION = "population"
CHANNELS = [f"Channel{i}" for i in range(5)]
IMPRESSION_COLS = [f"{c}_impression" for c in CHANNELS]
SPEND_COLS = [f"{c}_spend" for c in CHANNELS]
CONTROLS = ["competitor_sales_control", "sentiment_score_control"]
NON_MEDIA_TREATMENTS = ["Promo"]
ORGANIC_MEDIA = ["Organic_channel0_impression"]

SEED = 20260128
GATES_CONFIG = ROOT / "config" / "gates.toml"
MODEL_CONFIG = ROOT / "config" / "model.toml"


def load_model_config() -> dict:
  """The production model's specification, sampler and paths (config/model.toml)."""
  import tomllib
  with open(MODEL_CONFIG, "rb") as f:
    return tomllib.load(f)


def model_dir(kind: str = "model", train_weeks: int | None = None) -> Path:
  """Directory of the production fit ('model'), a CV refit, or the calibrated refit."""
  paths = load_model_config()["paths"]
  if train_weeks is not None:
    return ROOT / paths["cv"] / f"train{train_weeks}"
  return ROOT / paths[kind]


def load_model(kind: str = "model", train_weeks: int | None = None):
  """Loads a saved fit with meridian_serde (skill convention)."""
  from meridian.schema.serde import meridian_serde
  return meridian_serde.load_meridian(str(model_dir(kind, train_weeks) / "meridian_model.binpb"))


def load_gates_config() -> dict:
  """Gate rules and thresholds (config/gates.toml)."""
  import tomllib
  with open(GATES_CONFIG, "rb") as f:
    return tomllib.load(f)

# Categorical palette (dataviz reference instance, light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948"]
CHANNEL_COLOR = dict(zip(CHANNELS, PALETTE))
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def load_raw() -> pd.DataFrame:
  """Reads the raw CSV after verifying its checksum. Never modifies it."""
  if sha256(RAW_CSV) != DATA_SHA256:
    raise RuntimeError(f"Checksum mismatch for {RAW_CSV}: the raw file has been changed.")
  df = pd.read_csv(RAW_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
  df[TIME] = pd.to_datetime(df[TIME])
  df["revenue"] = df[KPI] * df[REVENUE_PER_KPI]
  return df.sort_values([GEO, TIME]).reset_index(drop=True)


def build_input_data(df: pd.DataFrame | None = None):
  """Builds Meridian InputData with the approved mapping (skill convention)."""
  from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder

  if df is None:
    df = pd.read_csv(RAW_CSV).drop(columns=["Unnamed: 0"], errors="ignore")
  builder = DataFrameInputDataBuilder(kpi_type="non_revenue")
  builder.with_kpi(df, kpi_col=KPI, time_col=TIME, geo_col=GEO)
  builder.with_revenue_per_kpi(df, revenue_per_kpi_col=REVENUE_PER_KPI)
  builder.with_population(df, population_col=POPULATION, geo_col=GEO)
  builder.with_controls(df, control_cols=CONTROLS)
  builder.with_non_media_treatments(df, non_media_treatment_cols=NON_MEDIA_TREATMENTS)
  builder.with_media(df, media_cols=IMPRESSION_COLS, media_spend_cols=SPEND_COLS,
                     media_channels=CHANNELS)
  builder.with_organic_media(df, organic_media_cols=ORGANIC_MEDIA)
  return builder.build()


def style_matplotlib():
  import matplotlib as mpl
  mpl.use("Agg")
  mpl.rcParams.update({
      "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
      "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": MUTED,
      "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
      "grid.linewidth": 0.6, "axes.spines.top": False,
      "axes.spines.right": False, "axes.prop_cycle": mpl.cycler(color=PALETTE),
      "lines.linewidth": 2, "font.size": 9, "axes.titlesize": 10,
      "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
      "savefig.dpi": 150, "savefig.bbox": "tight",
  })


def to_jsonable(o):
  if isinstance(o, dict):
    return {str(k): to_jsonable(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [to_jsonable(v) for v in o]
  if isinstance(o, (np.floating, float)):
    return None if not np.isfinite(o) else float(o)
  if isinstance(o, np.integer):
    return int(o)
  if isinstance(o, np.bool_):
    return bool(o)
  if isinstance(o, np.ndarray):
    return to_jsonable(o.tolist())
  return o


def write_json(obj, path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(to_jsonable(obj), f, indent=2)
    f.write("\n")


# --- Convergence gate (rules and thresholds: config/gates.toml) ---------------
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


# --- Predictions of a fitted model -------------------------------------------
def revenue_draws(mmm, use_posterior=True):
  """Expected and predictive revenue draws, (chain, draw, geo, time)."""
  from meridian import backend
  from meridian.analysis.analyzer import Analyzer
  an = Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
  mean_scaled = np.asarray(an.expected_outcome(
      use_posterior=use_posterior, use_kpi=True, aggregate_geos=False,
      aggregate_times=False, inverse_transform_outcome=False))
  group = mmm.inference_data.posterior if use_posterior else mmm.inference_data.prior
  sigma = np.asarray(group["sigma"])
  sigma = sigma[..., None, None] if sigma.ndim == 2 else sigma[..., None]
  rng = np.random.default_rng(1 if use_posterior else 2)
  pred_scaled = mean_scaled + sigma * rng.standard_normal(mean_scaled.shape)
  tr = mmm.model_context.kpi_transformer
  inv = lambda x: np.asarray(tr.inverse(backend.to_tensor(x, dtype=backend.float_dtype)))
  rpk = np.asarray(mmm.input_data.revenue_per_kpi)
  return inv(mean_scaled) * rpk, inv(pred_scaled) * rpk
