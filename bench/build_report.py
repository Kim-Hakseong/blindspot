"""Assemble the report the viewer serves, from committed benchmark outputs only.

Nothing here measures detector accuracy; it gathers what the benches measured
and validates the result against `blindspot.report.Report`, so a report that
breaks R1, R2 or the not-measured rule cannot be written. The one thing
computed here is coverage, which needs only degradation and measurement (no
inference).

    uv run python bench/build_report.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.boundary.coverage import axis_coverage, native_measurements, population_sweep  # noqa: E402
from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.report import NOT_MEASURED, Report  # noqa: E402

OUT = ROOT / "bench" / "out"


def limitations_from_results(path: pathlib.Path) -> list[str]:
    """The numbered 'Known limitations' in docs/results.md, one line each.

    Parsed rather than retyped so the report and the document cannot drift.
    """
    text = path.read_text(encoding="utf-8")
    section = text.split("## Known limitations", 1)[1]
    items = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*", section, flags=re.MULTILINE)
    # Plain text for display: drop markdown code ticks and emphasis markers.
    return [re.sub(r"\s+", " ", item.replace("`", "")).strip() for item in items]


def findings(efficiency: dict, frames_index: list[dict], seed: int) -> list[dict]:
    frame_by_axis = {e["axis"]: e["frame"]["image_id"] for e in frames_index}
    out = []
    for entry in efficiency["axes"]:
        degradation, _, field = entry["axis"].partition(".")
        axis = REGISTRY.get(degradation).axis(field)
        level = max(entry["levels"], key=lambda l: l["grid_steps"])
        grid = level["grid"]
        located = grid["lower"] is not None
        fail_value = (grid["upper"] if axis.severe_end == "hi" else grid["lower"]) if located else axis.severe
        out.append({
            "axis": entry["axis"],
            "unit": entry["unit"],
            "status": "located" if located else "passes_throughout",
            "lower": grid["lower"],
            "upper": grid["upper"],
            "probes_used": level["bisection_verified"]["probes_used"],
            "reproduce": {
                "axis": entry["axis"],
                "value": fail_value,
                "unit": entry["unit"],
                "seed": seed,
                "source_frame": frame_by_axis.get(entry["axis"], "set-level"),
                "command": (f"uv run blindspot probe --degradation {degradation} "
                            f"--axis {field} --value {float(fail_value)!r} --seed {seed}"),
            },
        })
    return out


def coverage(frames, efficiency: dict, seed: int, steps: int) -> list[dict]:
    regions = []
    for entry in efficiency["axes"]:
        degradation, _, field = entry["axis"].partition(".")
        axis = REGISTRY.get(degradation).axis(field)
        if not axis.expect:
            continue
        metric = axis.expect[0]
        values = [float(v) for v in np.linspace(axis.lo, axis.hi, steps)]
        cov = axis_coverage(
            axis_id=entry["axis"], unit=axis.unit, axis_range=(axis.lo, axis.hi),
            metric=metric, sweep_values=values,
            sweep_metric=population_sweep(frames, degradation, field, values, metric, seed),
            native_metric_values=native_measurements(frames, metric),
        ).to_dict()
        regions.append({
            "axis": cov["axis"],
            "unit": cov["unit"],
            "coverage_percent": cov["coverage_percent"],
            "uncovered": [{"from": r["from"], "to": r["to"], "unit": r["unit"]}
                          for r in cov["uncovered_regions"]],
            "inferred_from": f"{metric} of undegraded frames vs population-median sweep",
        })
    return regions


def main() -> int:
    from blindspot.runner.dataset import ValidationSet
    from blindspot.runner.yolox import COCO_CLASSES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=ROOT / "val" / "road100")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "viewer" / "public" / "data")
    parser.add_argument("--coverage-steps", type=int, default=17)
    args = parser.parse_args()

    efficiency = json.loads((OUT / "efficiency.json").read_text())
    frames_index = json.loads((OUT / "frames" / "frames.json").read_text())
    seed = efficiency["seed"]
    vs = ValidationSet(args.dataset, COCO_CLASSES)
    loaded = vs.load()

    args.out.mkdir(parents=True, exist_ok=True)

    map2d = None
    hook = OUT / "hook_grid.json"
    if hook.is_file():
        map2d = json.loads(hook.read_text())
        cells_out = args.out / "cells"
        cells_out.mkdir(exist_ok=True)
        for row in map2d["cells"]:
            for cell in row:
                if cell["image"]:
                    src = OUT / "hook_cells" / cell["image"]
                    shutil.copy2(src, cells_out / cell["image"])
                    cell["detections"] = json.loads(src.with_suffix(".json").read_text())

    frames_out = args.out / "frames"
    frames_out.mkdir(exist_ok=True)
    for entry in frames_index:
        shutil.copy2(OUT / "frames" / entry["png"], frames_out / entry["png"])

    run_id = "local-" + subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip()
    report = Report.model_validate({
        "uncovered_regions": coverage(loaded, efficiency, seed, args.coverage_steps),
        "run": {
            "run_id": run_id,
            "pipeline": efficiency["pipeline"]["name"],
            "dataset": efficiency["dataset"]["name"],
            "frames": efficiency["dataset"]["frames"],
            "objects": efficiency["dataset"]["objects"],
            "seed": seed,
            "baseline_map50": efficiency["baseline"]["mAP50"],
            "threshold_map50": efficiency["criterion"]["threshold_map50"],
            "criterion": efficiency["criterion"]["rule"],
        },
        "findings": findings(efficiency, frames_index, seed),
        "map2d": map2d,
        "curves": [{"axis": e["axis"], "unit": e["unit"], "points": e["map50_curve"]}
                   for e in efficiency["axes"]],
        "efficiency": efficiency["summary"],
        "evidence_frames": frames_index,
        "measurements": {"sim2real_gap": NOT_MEASURED, "cool_vs_x86": NOT_MEASURED},
        "limitations": limitations_from_results(ROOT / "docs" / "results.md"),
    })
    (args.out / "report.json").write_text(report.dump_json() + "\n", encoding="utf-8")
    # Coverage is computed here, so it is also persisted where documents can
    # cite it (the viewer's copy is a build artefact and not committed).
    (OUT / "coverage.json").write_text(json.dumps({
        "command": "uv run python bench/build_report.py",
        "method": "undegraded frames' objective measure inverted through the population-median sweep",
        "regions": [r.model_dump(by_alias=True) for r in report.uncovered_regions],
        # ":" not "." in keys, so a document can cite them by dotted path.
        "by_axis": {r.axis.replace(".", ":"): r.coverage_percent for r in report.uncovered_regions},
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'report.json'} (map2d: {'yes' if map2d else 'not yet'})")
    for region in report.uncovered_regions:
        print(f"  {region.axis}: covers {region.coverage_percent:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
