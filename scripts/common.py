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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "geo_all_channels.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
MODEL_BUILD = ROOT / "model_build"

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
