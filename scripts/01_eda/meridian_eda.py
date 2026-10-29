"""Meridian's standard EDA checks for the production model (meridian-model-building skill).

Builds the model from config/model.toml, draws from the prior (no fitting), runs
Meridian's EDA, saves its HTML report to reports/meridian/eda.html and a summary of
each check to outputs/01_eda/meridian_eda.json for reports/eda.md.
"""

import numpy as np
from meridian.model.eda import meridian_eda
from meridian.model.model import Meridian

from common import OUTPUTS, ROOT, build_input_data, load_model_config, write_json
from model_spec import make_spec


def severity(outcome):
  """Worst severity among a check's findings: INFO (fine) or REVIEW (flagged)."""
  levels = [f.severity for f in outcome.findings]
  return max(levels, key=lambda s: s.value).name if levels else "INFO"


def data_adequacy(engine):
  a = engine.check_data_param_ratio().analysis_artifacts[0]
  n_data = a.n_geos * a.n_times
  # Meridian's rough guide: data points per parameter, counting one parameter per geo
  # (minus the baseline geo), per knot, per control and per treatment.
  n_params = (a.n_geos - 1) + a.n_knots + a.n_controls + a.n_treatments
  return {"n_data_points": int(n_data), "n_parameters": int(n_params),
          "ratio": n_data / n_params, "n_knots": int(a.n_knots)}


def cost_per_media_unit(engine):
  outcome = engine.check_cost_per_media_unit()
  a = outcome.analysis_artifacts[0]
  cpm = a.cost_per_media_unit_da * 1000  # cost per 1,000 impressions
  per_channel = {}
  for channel in cpm.channel.values:
    values = cpm.sel(channel=channel).values
    values = values[~np.isnan(values)]  # weeks with no impressions have no CPM
    per_channel[str(channel)] = {"min": float(values.min()), "median": float(np.median(values)),
                                 "max": float(values.max())}
  return {"severity": severity(outcome), "n_outliers_flagged": int(len(a.outlier_df)),
          "cpm": per_channel}


def correlation(engine):
  outcome = engine.check_pairwise_corr()
  a = outcome.get_overall_artifacts()[0]
  corr = a.corr_matrix
  names = [str(v) for v in corr.var1.values]
  values = np.abs(np.asarray(corr.values))
  np.fill_diagonal(values, 0)          # ignore each variable's correlation with itself
  i, j = np.unravel_index(values.argmax(), values.shape)
  return {"severity": severity(outcome), "max_abs_corr": float(values[i, j]),
          "max_pair": [names[i], names[j]], "flag_threshold": float(a.extreme_corr_threshold),
          "n_pairs_flagged": int(len(a.extreme_corr_var_pairs))}


def vif(engine):
  outcome = engine.check_vif()
  a = outcome.get_overall_artifacts()[0]
  values = {str(k): float(v) for k, v in zip(a.vif_da["var"].values, a.vif_da.values)}
  worst = max(values, key=values.get)
  return {"severity": severity(outcome), "vif": values, "max_vif": values[worst], "max_vif_variable": worst}


def variability(engine):
  outcome = engine.check_std()
  outliers = {a.variable: int(len(a.outlier_df)) for a in outcome.analysis_artifacts}
  kpi = engine.check_overall_kpi_invariability()
  return {"severity": severity(outcome), "outliers_flagged": outliers,
          "kpi_varies": severity(kpi) == "INFO",
          "kpi_scaled_sd": float(kpi.analysis_artifacts[0].kpi_stdev)}


def population(engine):
  raw = engine.check_population_corr_raw_media().analysis_artifacts[0].correlation_ds
  scaled = engine.check_population_corr_scaled_treatment_control().analysis_artifacts[0].correlation_ds
  media = {str(c): float(v) for c, v in zip(raw.media_channel.values, raw["media"].values)}
  other = {str(c): float(v) for c, v in zip(scaled.control_variable.values, scaled["controls_scaled"].values)}
  other.update({str(c): float(v) for c, v in zip(scaled.non_media_channel.values,
                                                  scaled["non_media_treatments_scaled"].values)})
  return {"raw_media_spearman": media, "controls_and_promo_spearman": other}


def time_vs_geo(engine):
  r2 = engine.check_variable_geo_time_collinearity().analysis_artifacts[0].rsquared_ds
  names = [str(v) for v in r2["var"].values]
  # Share of each variable's variation explained by the week alone, and by the geo alone.
  by_time = dict(zip(names, map(float, r2["rsquared_time"].values)))
  by_geo = dict(zip(names, map(float, r2["rsquared_geo"].values)))
  worst = max(by_time, key=by_time.get)
  return {"r2_time": by_time, "r2_geo": by_geo, "max_r2_time": by_time[worst],
          "max_r2_time_variable": worst, "max_r2_geo": max(by_geo.values())}


def main():
  cfg = load_model_config()["sampler"]
  data = build_input_data()
  mmm = Meridian(input_data=data, model_spec=make_spec(data))
  mmm.sample_prior(n_draws=cfg["n_prior_draws"], seed=cfg["seed"])

  eda = meridian_eda.MeridianEDA(mmm)
  report_dir = ROOT / "reports" / "meridian"
  eda.generate_and_save_report(filename="eda.html", filepath=str(report_dir))

  engine = eda.eda_engine
  summary = {
      "data_adequacy": data_adequacy(engine),
      "cost_per_media_unit": cost_per_media_unit(engine),
      "correlation": correlation(engine),
      "vif": vif(engine),
      "variability": variability(engine),
      "population": population(engine),
      "time_vs_geo": time_vs_geo(engine),
      "report": "reports/meridian/eda.html",
  }
  write_json(summary, OUTPUTS / "01_eda" / "meridian_eda.json")
  print(f"Wrote {report_dir / 'eda.html'} and outputs/01_eda/meridian_eda.json")


if __name__ == "__main__":
  main()
