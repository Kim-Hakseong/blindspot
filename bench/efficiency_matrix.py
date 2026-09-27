"""Collect the probe-savings runs across datasets and pipelines (W3-5).

Reads every bench/out/efficiency*.json and writes one matrix: per dataset x
pipeline x axis, the grid's boundary, the verified bisection's status, its
error against the grid and its savings, at the finest precision measured.
Nothing is re-measured here.

    uv run python bench/efficiency_matrix.py
"""

from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def main() -> int:
    rows = []
    for path in sorted(p for p in OUT.glob("efficiency*.json") if p.name != "efficiency_matrix.json"):
        d = json.loads(path.read_text())
        for axis in d["axes"]:
            level = max(axis["levels"], key=lambda l: l["grid_steps"])
            v = level["bisection_verified"]
            rows.append({
                "dataset": d["dataset"]["name"], "frames": d["dataset"]["frames"],
                "pipeline": d["pipeline"]["name"], "baseline_map50": d["baseline"]["mAP50"],
                "axis": axis["axis"], "unit": axis["unit"],
                "grid_steps": level["grid_steps"],
                "boundary": [level["grid"]["lower"], level["grid"]["upper"]],
                "status": v["status"], "probes_verified": v["probes_used"],
                "boundary_error": v["boundary_error"], "savings_verified": v["savings"],
                "source": path.name,
            })
    combos = sorted({(r["dataset"], r["pipeline"]) for r in rows})
    valid = [r for r in rows if r["savings_verified"]]
    summary = {
        "combinations": len(combos),
        "axis_runs": len(rows),
        "located": sum(r["status"] == "located" for r in rows),
        "valid_savings": len(valid),
        "max_boundary_error": max((r["boundary_error"] or 0.0) for r in rows) if rows else None,
        "min_savings_verified": min(r["savings_verified"] for r in valid) if valid else None,
    }
    (OUT / "efficiency_matrix.json").write_text(json.dumps(
        {"summary": summary, "combinations": [list(c) for c in combos], "rows": rows,
         "command": "uv run python bench/efficiency_matrix.py"}, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
