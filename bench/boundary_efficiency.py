"""Probe savings: active boundary search against an exhaustive grid.

This is the project's headline measurement and the only place the savings
figure is produced. Nothing else may state a savings number; documentation
cites this output and `tools/check_doc_numbers.py` verifies the citation.

The comparison is only meaningful if it is honest about three things, so all
three are enforced here:

1. **Same everything.** Same dataset, pipeline, seed and failure criterion for
   both strategies. Only the choice of which conditions to probe differs.
2. **Savings require agreement.** A search that uses fewer probes but finds a
   different boundary has not saved anything. Savings are reported as valid
   only when the active boundary lands within one grid cell of the grid's.
3. **Both modes reported.** Verified bisection costs more probes than cheap
   bisection and catches non-monotone responses the cheap mode misses. Quoting
   only the cheap number would be quoting the blinder method.

    uv run python bench/boundary_efficiency.py --out bench/out/efficiency.json
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
from blindspot.boundary.locate import (  # noqa: E402
    BoundaryStatus,
    locate_on_axis,
    scan_boundaries,
)
from blindspot.boundary.probe import baseline_map50, run_probe  # noqa: E402
from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.runner.dataset import ValidationSet  # noqa: E402
from blindspot.runner.registry import PIPELINES, load_pipeline  # noqa: E402
from blindspot.runner.yolox import COCO_CLASSES  # noqa: E402

DEFAULT_AXES = (
    "motion_blur.exposure_ms",
    "low_light.illuminance_lux",
    "fog.beta_per_m",
    "jpeg.quality",
)


class ProbeStore:
    """Disk-backed mAP@50 for a probe, keyed by everything that determines it.

    A probe is deterministic in `(dataset, frame count, pipeline, degradation,
    axis, value, seed)`, so running one twice buys nothing. Persisting them
    makes a long comparison resumable and keeps a re-run from paying again for
    work already done -- which is the same argument the tool makes about cloud
    spend, applied to itself.

    This does not distort the comparison: probe *counts* are attributed per
    strategy by the evaluators below, independently of whether a value was
    served from disk.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data: dict[str, float] = {}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(dataset, n_frames, pipeline, degradation, axis, value, seed) -> str:
        return f"{dataset}|{n_frames}|{pipeline}|{degradation}|{axis}|{value:.9g}|{seed}"

    def get(self, key: str) -> float | None:
        return self.data.get(key)

    def put(self, key: str, map50: float) -> None:
        self.data[key] = map50
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True), encoding="utf-8")


class CachedEvaluator:
    """Runs probes and counts how many distinct conditions this strategy needed.

    Each strategy gets a fresh evaluator, so its probe count reflects only its
    own choices. The shared ProbeStore avoids recomputing a condition another
    strategy already measured, without crediting that saving to anyone.
    """

    def __init__(
        self, frames, pipeline, validation_set, criterion, degradation, axis, seed, store
    ):
        self.frames = frames
        self.pipeline = pipeline
        self.validation_set = validation_set
        self.criterion = criterion
        self.degradation = degradation
        self.axis = axis
        self.seed = seed
        self.store = store
        self.cache: dict[float, tuple[bool, float]] = {}

    def __call__(self, value: float) -> bool:
        key = round(float(value), 9)
        if key in self.cache:
            return self.cache[key][0]

        store_key = ProbeStore.key(
            self.validation_set.name,
            len(self.frames),
            self.pipeline.name,
            self.degradation,
            self.axis,
            float(value),
            self.seed,
        )
        map50 = self.store.get(store_key)
        if map50 is None:
            result = run_probe(
                self.frames,
                self.pipeline,
                self.validation_set,
                self.degradation,
                self.axis,
                float(value),
                self.seed,
            )
            map50 = result.map50
            self.store.put(store_key, map50)

        failed = self.criterion.failed(map50)
        self.cache[key] = (failed, map50)
        return failed

    @property
    def probes_used(self) -> int:
        return len(self.cache)


def _midpoint(interval) -> float | None:
    if interval is None:
        return None
    return (interval[0] + interval[1]) / 2.0


def evaluate_axis(axis_id, frames, pipeline, validation_set, criterion, levels, seed, store):
    """Measure savings as a function of the precision demanded.

    Savings is not a single number, because the two strategies scale
    differently: to halve the uncertainty in the boundary position, a grid must
    double its probe count while bisection needs one more probe. Quoting one
    figure without its precision target would be quoting an arbitrary point on
    a curve.

    Grids are sized 2^k+1 so that every coarser grid is a strict subset of the
    finest one. Each coarse level is therefore built from probes that were
    really run, not from an extrapolation of what a coarse grid would have
    cost.
    """
    degradation, _, axis_field = axis_id.partition(".")
    deg = REGISTRY.get(degradation)
    axis = deg.axis(axis_field)

    def make_evaluator():
        return CachedEvaluator(
            frames, pipeline, validation_set, criterion, degradation, axis_field, seed, store
        )

    # ---- Run the finest grid once; coarser levels are subsets of it ------
    finest = max(levels)
    grid = make_evaluator()
    started = time.perf_counter()
    # Sweep from benign to severe so the transition reads as pass -> fail.
    finest_values = [axis.from_severity(s) for s in np.linspace(0.0, 1.0, finest)]
    finest_samples = [(float(v), grid(float(v))) for v in finest_values]
    grid_seconds = time.perf_counter() - started

    entries = []
    for steps in sorted(levels):
        cell = (axis.hi - axis.lo) / (steps - 1)
        stride = (finest - 1) // (steps - 1)
        samples = finest_samples[::stride]
        assert len(samples) == steps, f"level {steps} is not nested in {finest}"

        # Samples are already in benign -> severe order; scan in that order.
        grid_transitions = scan_boundaries(
            [(axis.to_severity(v), f) for v, f in samples]
        )
        grid_transitions = [
            tuple(sorted((axis.from_severity(a), axis.from_severity(b))))
            for a, b in grid_transitions
        ]
        grid_interval = grid_transitions[0] if grid_transitions else None
        grid_mid = _midpoint(grid_interval)

        cheap_eval = make_evaluator()
        t0 = time.perf_counter()
        cheap_result = locate_on_axis(axis, cheap_eval, target_width=cell)
        cheap_seconds = time.perf_counter() - t0

        verified_eval = make_evaluator()
        t0 = time.perf_counter()
        verified_result = locate_on_axis(
            axis, verified_eval, target_width=cell, verify_samples=5
        )
        verified_seconds = time.perf_counter() - t0

        def compare(result, evaluator, seconds):
            error = None
            valid = False
            if (
                result.status is BoundaryStatus.LOCATED
                and grid_mid is not None
                and result.midpoint is not None
            ):
                error = abs(result.midpoint - grid_mid)
                valid = error <= cell
            return {
                "status": result.status.value,
                "lower": result.lower,
                "upper": result.upper,
                "midpoint": result.midpoint,
                "probes_used": evaluator.probes_used,
                "wall_seconds": round(seconds, 2),
                "boundary_error": error,
                "boundary_error_in_grid_cells": (error / cell) if error is not None else None,
                "savings_valid": valid,
                "savings": (steps / evaluator.probes_used) if valid else None,
                "n_transitions": len(result.transitions),
            }

        entries.append(
            {
                "grid_steps": steps,
                "precision": cell,
                "precision_unit": axis.unit,
                "grid": {
                    "probes_used": steps,
                    "lower": grid_interval[0] if grid_interval else None,
                    "upper": grid_interval[1] if grid_interval else None,
                    "midpoint": grid_mid,
                    "n_transitions": len(grid_transitions),
                },
                "bisection_cheap": compare(cheap_result, cheap_eval, cheap_seconds),
                "bisection_verified": compare(verified_result, verified_eval, verified_seconds),
            }
        )

    return {
        "axis": axis_id,
        "unit": axis.unit,
        "range": [axis.lo, axis.hi],
        "finest_grid_steps": finest,
        "finest_grid_wall_seconds": round(grid_seconds, 2),
        "map50_curve": [
            {"value": v, "map50": round(grid.cache[round(v, 9)][1], 6)}
            for v, _ in finest_samples
        ],
        "levels": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("val/road100"))
    parser.add_argument("--pipeline", default="yolox_s", choices=sorted(PIPELINES))
    parser.add_argument("--axes", nargs="*", default=list(DEFAULT_AXES))
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=[5, 9, 17, 33],
        help="Grid sizes to compare at, each 2^k+1 so coarse grids nest inside fine ones",
    )
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/efficiency.json"))
    parser.add_argument(
        "--probe-cache",
        type=pathlib.Path,
        default=pathlib.Path(".cache/probes.json"),
        help="Disk cache of probe results, so a long comparison is resumable",
    )
    args = parser.parse_args()

    validation_set = ValidationSet(args.dataset, COCO_CLASSES)
    frames = validation_set.load(limit=args.frames)
    pipeline = load_pipeline(args.pipeline)

    store = ProbeStore(args.probe_cache)
    base = baseline_map50(frames, pipeline, validation_set)
    criterion = Criterion(baseline_map50=base["mAP50"])
    print(f"{validation_set.name}: {len(frames)} frames, "
          f"{sum(len(f.truths) for f in frames)} objects")
    print(f"baseline mAP@50 {base['mAP50']:.4f}, failure threshold {criterion.threshold:.4f}\n")

    entries = []
    for axis_id in args.axes:
        print(f"--- {axis_id}")
        entry = evaluate_axis(
            axis_id, frames, pipeline, validation_set, criterion, args.levels, args.seed,
            store,
        )
        entries.append(entry)
        print(f"    {'grid':>6} {'precision':>12}   {'cheap':>16} {'verified':>16}")
        for level in entry["levels"]:
            def fmt(b):
                if not b["savings"]:
                    return f"{b['probes_used']:>3}p {b['status'][:9]:>11}"
                return f"{b['probes_used']:>3}p {b['savings']:>10.2f}x"

            print(f"    {level['grid_steps']:>6} "
                  f"{level['precision']:>9.4g} {entry['unit']:<3} "
                  f"{fmt(level['bisection_cheap']):>16} "
                  f"{fmt(level['bisection_verified']):>16}")

    # Savings depends on the precision demanded, so it is summarised per level
    # rather than collapsed into one figure.
    by_level: dict[int, dict] = {}
    for steps in sorted(args.levels):
        cheap, verified = [], []
        for entry in entries:
            level = next(l for l in entry["levels"] if l["grid_steps"] == steps)
            if level["bisection_cheap"]["savings"]:
                cheap.append(level["bisection_cheap"]["savings"])
            if level["bisection_verified"]["savings"]:
                verified.append(level["bisection_verified"]["savings"])
        by_level[steps] = {
            "grid_steps": steps,
            "axes_with_valid_savings_cheap": len(cheap),
            "axes_with_valid_savings_verified": len(verified),
            "mean_savings_cheap": round(float(np.mean(cheap)), 4) if cheap else None,
            "mean_savings_verified": round(float(np.mean(verified)), 4) if verified else None,
            "min_savings_verified": round(float(np.min(verified)), 4) if verified else None,
        }

    finest = max(args.levels)
    finest_summary = by_level[finest]
    summary = {
        "axes_evaluated": len(entries),
        "note": (
            "Savings scales with the precision demanded: a grid must double its "
            "probes to halve boundary uncertainty, bisection needs one more. "
            "The gate is evaluated at the finest precision measured."
        ),
        "by_grid_level": by_level,
        "gate_threshold": 3.0,
        "gate_grid_steps": finest,
        "gate_metric": "mean_savings_verified",
        "gate_value": finest_summary["mean_savings_verified"],
        "gate_passed": bool(
            finest_summary["mean_savings_verified"]
            and finest_summary["mean_savings_verified"] >= 3.0
        ),
    }

    report = {
        "dataset": {
            "name": validation_set.name,
            "frames": len(frames),
            "objects": sum(len(f.truths) for f in frames),
            "source": validation_set.manifest["source"],
        },
        "pipeline": pipeline.describe(),
        "criterion": criterion.describe(),
        "baseline": base,
        "grid_levels": sorted(args.levels),
        "seed": args.seed,
        "summary": summary,
        "axes": entries,
        "environment": {
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "command": "uv run python bench/boundary_efficiency.py",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'grid':>6} {'mean savings cheap':>20} {'mean savings verified':>23}")
    for steps, level in by_level.items():
        cheap = f"{level['mean_savings_cheap']:.2f}x" if level["mean_savings_cheap"] else "n/a"
        verified = (
            f"{level['mean_savings_verified']:.2f}x"
            if level["mean_savings_verified"] else "n/a"
        )
        print(f"{steps:>6} {cheap:>20} {verified:>23}")
    print(f"\nW3 gate (>= 3x, verified, at grid={finest}): "
          f"{'PASS' if summary['gate_passed'] else 'FAIL'}"
          f"  measured={summary['gate_value']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
