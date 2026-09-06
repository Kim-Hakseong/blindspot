"""Exhaustive grid sweep of a condition axis -- the control that active search
must beat.

This is the honest baseline. Every claim about probe savings is measured
against these numbers, on the same dataset, the same pipeline and the same
failure criterion.

    uv run python bench/grid_baseline.py --axis motion_blur.exposure_ms --steps 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from blindspot.boundary.criterion import Criterion  # noqa: E402
from blindspot.boundary.probe import baseline_map50, run_probe  # noqa: E402
from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.runner.dataset import ValidationSet  # noqa: E402
from blindspot.runner.yolox import COCO_CLASSES, YoloxPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", default="motion_blur.exposure_ms")
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("val/road100"))
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("models/yolox_s.onnx"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=None, help="limit frames (for a quick look)")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    degradation, _, axis_field = args.axis.partition(".")
    deg = REGISTRY.get(degradation)
    axis = deg.axis(axis_field)

    validation_set = ValidationSet(args.dataset, COCO_CLASSES)
    frames = validation_set.load(limit=args.frames)
    if not frames:
        sys.exit(f"no frames loaded from {args.dataset}")
    pipeline = YoloxPipeline(args.model)

    print(f"dataset {validation_set.name}: {len(frames)} frames, "
          f"{sum(len(f.truths) for f in frames)} labelled objects")
    print(f"pipeline {pipeline.name} via cv::dnn")

    started = time.perf_counter()
    base = baseline_map50(frames, pipeline, validation_set)
    criterion = Criterion(baseline_map50=base["mAP50"])
    print(f"undegraded baseline mAP@50 = {base['mAP50']:.4f}")
    print(f"failure threshold = {criterion.threshold:.4f} "
          f"({criterion.fraction:.0%} of baseline)\n")

    print(f"{'value':>10} {axis.unit:<10} {'mAP@50':>8} {'verdict':>8}   derived")
    print("-" * 78)

    probes = []
    for value in np.linspace(axis.lo, axis.hi, args.steps):
        result = run_probe(
            frames, pipeline, validation_set, degradation, axis_field, float(value), args.seed
        )
        probes.append(result)
        verdict = "FAIL" if criterion.failed(result.map50) else "pass"
        derived = " ".join(
            f"{k}={v['value']:.3g}{v['unit']}" for k, v in result.derived.items()
        )
        print(f"{value:>10.4g} {axis.unit:<10} {result.map50:>8.4f} {verdict:>8}   {derived}")

    elapsed = time.perf_counter() - started

    # Locate the pass -> fail transition as an interval, never a point.
    boundary = None
    for previous, current in zip(probes, probes[1:]):
        if not criterion.failed(previous.map50) and criterion.failed(current.map50):
            boundary = {
                "axis": args.axis,
                "unit": axis.unit,
                "lower": previous.value,
                "upper": current.value,
                "width": current.value - previous.value,
                "map50_before": previous.map50,
                "map50_after": current.map50,
                "reproduce": current.command,
            }
            break

    report = {
        "strategy": "grid",
        "axis": args.axis,
        "unit": axis.unit,
        "range": [axis.lo, axis.hi],
        "steps": args.steps,
        "probes_used": len(probes),
        "seed": args.seed,
        "dataset": {
            "name": validation_set.name,
            "frames": len(frames),
            "objects": sum(len(f.truths) for f in frames),
            "source": validation_set.manifest["source"],
        },
        "pipeline": pipeline.describe(),
        "criterion": criterion.describe(),
        "baseline": base,
        "boundary": boundary,
        "wall_seconds": round(elapsed, 2),
        "environment": {
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "command": (
            f"uv run python bench/grid_baseline.py --axis {args.axis} --steps {args.steps}"
        ),
        "probes": [p.to_dict() for p in probes],
    }

    print(f"\n{len(probes)} probes in {elapsed:.1f}s "
          f"({elapsed / len(probes):.1f}s per probe)")
    if boundary:
        print(
            f"boundary: {boundary['lower']:.4g} -> {boundary['upper']:.4g} {axis.unit} "
            f"(width {boundary['width']:.4g}), "
            f"mAP@50 {boundary['map50_before']:.3f} -> {boundary['map50_after']:.3f}"
        )
    else:
        print("no pass->fail transition inside this range")

    out = args.out or pathlib.Path(f"bench/out/grid_{degradation}_{axis_field}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
