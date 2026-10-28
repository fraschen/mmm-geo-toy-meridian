"""Generates reports/gates.md: every gate's rule, threshold, rationale and latest result.

Rules come from config/gates.toml; results from outputs/gates/gates.json,
outputs/selection/selection.json and outputs/meridian/health_check.json.
"""

from __future__ import annotations

import json

from common import OUTPUTS, ROOT, load_gates_config

J = lambda p: json.load(open(OUTPUTS / p))


def fmt(v):
  if isinstance(v, float):
    return f"{v:.4g}"
  if isinstance(v, dict):
    return ", ".join(f"{k} {fmt(x)}" for k, x in v.items()
                     if not isinstance(x, dict) and not (isinstance(x, list) and not x))
  if isinstance(v, list):
    return "[" + ", ".join(fmt(x) for x in v) + "]"
  return str(v)


def main():
  cfg = load_gates_config()
  gates = J("gates/gates.json")
  sel = J("selection/selection.json")
  health = J("meridian/health_check.json")
  nf = J("eda/eda_summary.json")["noise_floor"]
  results = {g["gate"]: g for g in gates["gates"]}
  fill = dict(nat_floor=nf["national_mape_vs_13wk_ma"], nat_cv=nf["national_weekly_cv"],
              nat_ac=nf["national_lag1_autocorr"], geo_cv=nf["geo_weekly_cv"]["median"],
              geo_floor=nf["geo_mape_vs_13wk_ma"]["median"],
              ratio=cfg["gates"]["cv"]["thresholds"]["cv_national_mape_max"] / nf["national_mape_vs_13wk_ma"])

  summary = ["| gate | blocking | latest status |", "|---|---|---|"]
  body = []
  for name, g in cfg["gates"].items():
    hits = [r for k, r in results.items() if k == name or k.startswith(f"{name}_")]
    status = ", ".join(f"{r['gate']} {r['status']}" for r in hits) or "-"
    summary.append(f"| [{name}](#{name.replace('_', '-')}) | {'yes' if g['blocking'] else 'no (warn only)'} | {status} |")
    th = "\n".join(f"| `{k}` | {fmt(v)} |" for k, v in g["thresholds"].items()) or "| (none) | |"
    rows = "\n".join(f"| {r['gate']} | {r['status']} | {fmt(r['value'])} |" for r in hits
                     if not isinstance(r["value"], dict) or len(r["value"]) < 8)
    if name == "channel_roi_checks" and hits:
      rows = "\n".join(f"| {c} | {v['status']} | contraction {v['roi_contraction']:.2f}, P(ROI>1) {v['p_roi_gt_1']:.2f} |"
                       for c, v in hits[0]["value"].items())
    body += [f"### {name}", "",
             f"- **Blocking:** {'yes' if g['blocking'] else 'no (WARN only)'}",
             f"- **Applies to:** {g['applies_to']}",
             f"- **Rule:** {g['checks']}",
             f"- **Source of threshold:** {g['source']}",
             f"- **Basis:** {g['basis'].format(**fill)}",
             f"- **Enforced in:** `{g['enforced_in']}`", "",
             "| threshold | value |", "|---|---|", th, ""]
    if rows:
      body += ["Latest result:", "", "| check | status | value |", "|---|---|---|", rows, ""]
    notes = [r["note"] for r in hits if r.get("note")]
    body += [f"Note: {n}" for n in dict.fromkeys(notes)] + ([""] if notes else [])

  s = cfg["selection"]
  steps = "; ".join(f"{sel['labels'][x['challenger']]} vs {sel['labels'][x['current']]}: ELPD "
                    f"{x['elpd_diff']:+.1f} (paired SE {x['se']:.1f}, needs > {x['threshold']:.1f}) -> "
                    f"{'switch' if x['switch'] else 'keep'}" for x in sel["comparison_steps"])
  m = cfg["meridian_health"]
  hrows = "\n".join(f"| {c['check']} | {c['status']} |" for c in health["checks"])
  lines = [
      "# Gates",
      "",
      "Automated acceptance checks for the selected model. Rules and thresholds live in "
      "[`config/gates.toml`](../config/gates.toml), which the scripts read; this page is generated "
      "from it and from the latest outputs (`make gates`, then `make reports`). "
      "`make gates` exits non-zero if any blocking gate fails.",
      "",
      f"- **PASS:** the check meets its threshold. **WARN:** a non-blocking check misses it "
      f"(report with caveats). **FAIL:** a blocking check misses it (not client-ready).",
      f"- Intervals: {cfg['meta']['interval']}.",
      f"- {cfg['meta']['note']}",
      "",
      "## Summary",
      "",
      f"Selected model: `{gates['selected']}` ({gates['selection_basis']}). Blocking failures: "
      f"{', '.join(gates['blocking_failures']) or 'none'}.",
      "",
      *summary,
      "",
      "## Model selection rule",
      "",
      f"- **Applies to:** {s['applies_to']}",
      f"- **Rule:** {s['checks']}",
      f"- **Threshold:** `se_rule` = {s['thresholds']['se_rule']} ({s['source']}). {s['basis']}",
      f"- **Enforced in:** `{s['enforced_in']}`",
      f"- **Latest result:** {steps or sel['selection_basis']}.",
      "",
      "## Gate definitions and latest results",
      "",
      *body,
      "## Meridian's own health checks (not gating)",
      "",
      f"- {m['checks']}",
      f"- **Looser than our gates:** {m['looser_than_ours']}",
      f"- **Latest:** {health['overall_status']}, health score {health['health_score']:.1f}/100.",
      "",
      "| check | status |", "|---|---|", hrows, "",
  ]
  (ROOT / "reports" / "gates.md").write_text("\n".join(lines))
  print("Wrote reports/gates.md")


if __name__ == "__main__":
  main()
