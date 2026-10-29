"""Builds the "Acceptance gates" part of reports/results.md.

Rules and thresholds come from config/gates.toml; the latest results come from
outputs/03_validate/gates.json. This only documents the gates, it does not run them.
"""

import json

from common import OUTPUTS, load_gates_config


def load_json(path):
  with open(OUTPUTS / path) as f:
    return json.load(f)


def fmt(value):
  """Short text for a result value (a number, a list or a dict)."""
  if isinstance(value, float):
    return f"{value:.4g}"
  if isinstance(value, list):
    return "[" + ", ".join(fmt(v) for v in value) + "]"
  if isinstance(value, dict):
    parts = []
    for key, v in value.items():
      if isinstance(v, dict) or v == []:
        continue  # keep the table readable
      parts.append(f"{key} {fmt(v)}")
    return ", ".join(parts)
  return str(value)


def basis_values(cfg):
  """Numbers from the EDA that the 'basis' texts in gates.toml refer to."""
  noise = load_json("01_eda/eda_summary.json")["noise_floor"]
  mape_max = cfg["gates"]["cv"]["thresholds"]["cv_national_mape_max"]
  return {
      "nat_floor": noise["national_mape_vs_13wk_ma"],
      "nat_cv": noise["national_weekly_cv"],
      "nat_ac": noise["national_lag1_autocorr"],
      "geo_cv": noise["geo_weekly_cv"]["median"],
      "geo_floor": noise["geo_mape_vs_13wk_ma"]["median"],
      "ratio": mape_max / noise["national_mape_vs_13wk_ma"],
  }


def results_for(name, results):
  """Results of one gate. The cv gate has several (cv_fold_104, cv_holdout, ...)."""
  return [r for r in results if r["gate"] == name or r["gate"].startswith(name + "_")]


def result_rows(name, found):
  """Table rows with the latest result(s) of one gate."""
  rows = []
  if name == "channel_roi_checks":
    # One row per channel instead of one big dictionary.
    for channel, v in found[0]["value"].items():
      value = f"contraction {v['roi_contraction']:.2f}, P(ROI>1) {v['p_roi_gt_1']:.2f}"
      rows.append(f"| {channel} | {v['status']} | {value} |")
  else:
    for r in found:
      rows.append(f"| {r['gate']} | {r['status']} | {fmt(r['value'])} |")
  return rows


def gate_details(name, gate, found, basis):
  """The text block for one gate."""
  lines = [
      f"### {name}",
      "",
      f"**{gate['plain']}**",
      "",
      f"- **Blocking:** {'yes' if gate['blocking'] else 'no (WARN only)'}",
      f"- **Applies to:** {gate['applies_to']}",
      f"- **Rule:** {gate['checks']}",
      f"- **Source of threshold:** {gate['source']}",
      f"- **Basis:** {gate['basis'].format(**basis)}",
      f"- **Enforced in:** `{gate['enforced_in']}`",
      "",
      "| threshold | value |",
      "|---|---|",
  ]
  for key, value in gate["thresholds"].items():
    lines.append(f"| `{key}` | {fmt(value)} |")
  if not gate["thresholds"]:
    lines.append("| (none) | |")
  lines.append("")

  if found:
    lines += ["Latest result:", "", "| check | status | value |", "|---|---|---|"]
    lines += result_rows(name, found)
    lines.append("")
    # Notes are the same for several results (e.g. every failed fold), so show each once.
    notes = {r["note"] for r in found if r.get("note")}
    for note in sorted(notes):
      lines += [f"Note: {note}", ""]
  return lines


def section():
  """Returns the lines of the 'Acceptance gates' part."""
  cfg = load_gates_config()
  gates = load_json("03_validate/gates.json")
  basis = basis_values(cfg)

  summary = [
      "| gate | the question it answers | blocking | latest status |",
      "|---|---|---|---|",
  ]
  details = []
  for name, gate in cfg["gates"].items():
    found = results_for(name, gates["gates"])
    status = ", ".join(f"{r['gate']} {r['status']}" for r in found) or "-"
    blocking = "yes" if gate["blocking"] else "no (warn only)"
    summary.append(f"| [{name}](#{name}) | {gate['plain']} | {blocking} | {status} |")
    details += gate_details(name, gate, found, basis)

  failures = ", ".join(gates["blocking_failures"]) or "none"
  return [
      "# Part 3: Acceptance gates",
      "",
      "These are the checks the model must pass before its numbers go to a client. A **blocking** "
      "check that fails means the model is not client-ready; a non-blocking check only adds a caveat. "
      "Rules and thresholds live in [`config/gates.toml`](../config/gates.toml), which the scripts "
      "read. `make gates` exits non-zero if any blocking gate fails.",
      "",
      "- **PASS:** the check meets its threshold. **WARN:** a non-blocking check misses it "
      "(report with caveats). **FAIL:** a blocking check misses it (not client-ready).",
      f"- Intervals: {cfg['meta']['interval']}.",
      "",
      "## Gate summary",
      "",
      f"Blocking failures: {failures}.",
      "",
      *summary,
      "",
      "## Gate definitions and latest results",
      "",
      *details,
  ]
