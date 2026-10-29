"""Model health check and Model Results Summary (meridian-result-visualization skill).

Usage (from the project root):
  .venv/bin/python scripts/03_validate/results_report.py
"""

import json
import os

from meridian.analysis import summarizer
from meridian.analysis.review import reviewer
from meridian.schema.serde import meridian_serde

MODEL_PATH = "models/model/meridian_model.binpb"  # production model (config/model.toml)
OUTPUT_DIR = "reports/meridian"
START_DATE, END_DATE = "2021-01-25", "2024-01-15"


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # 1. Load the model.
  model = meridian_serde.load_meridian(MODEL_PATH)

  # 2. Post-modeling health checks.
  model_reviewer = reviewer.ModelReviewer(
      model_context=model.model_context, inference_data=model.inference_data
  )
  review_summary = model_reviewer.run()
  review_summary.output_model_health_card(
      filename="health_check.html", filepath=OUTPUT_DIR
  )
  print(f"Health check written to {OUTPUT_DIR}/health_check.html")

  # Machine-readable copy of the check outcomes for the gate report.
  def _plain(v):
    if isinstance(v, dict):
      return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
      return [_plain(x) for x in v]
    if hasattr(v, "item"):
      try:
        return v.item()
      except (ValueError, AttributeError):
        return str(v)
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
  checks = [{"check": type(r).__name__.replace("CheckResult", ""),
             "status": r.case.status.name, "details": _plain(dict(r.details)),
             "recommendation": getattr(r, "recommendation", "")}
            for r in review_summary.results]
  os.makedirs("outputs/03_validate", exist_ok=True)
  with open("outputs/03_validate/health_check.json", "w") as f:
    json.dump({"overall_status": review_summary.overall_status.name,
               "health_score": float(review_summary.health_score),
               "summary_message": review_summary.summary_message,
               "checks": checks}, f, indent=2, default=str)

  if os.environ.get("HEALTH_ONLY"):
    return

  # 3. Model Results Summary (revenue units: use_kpi=False).
  mmm_summarizer = summarizer.Summarizer(model, use_kpi=False)
  mmm_summarizer.output_model_results_summary(
      filename="results_summary.html",
      filepath=OUTPUT_DIR,
      start_date=START_DATE,
      end_date=END_DATE,
  )
  print(f"Results summary written to {OUTPUT_DIR}/results_summary.html")


if __name__ == "__main__":
  main()
