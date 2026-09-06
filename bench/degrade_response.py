"""Measure what each degradation axis objectively does to an image.

This is the source of every W1 number that appears in the documentation. It
sweeps each registered axis across its declared physical range and records the
objective measurements, so a claim like "MTF50 halves by 18 ms of exposure" has
a command behind it rather than an author behind it.

    uv run python bench/degrade_response.py --out bench/out/degrade_response.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.measure import measure  # noqa: E402

SEED = 20260906


def reference_scene(size: int = 256, seed: int = SEED) -> np.ndarray:
    """The same deterministic scene the test suite uses.

    Kept identical to tests/conftest.py so bench numbers and test assertions
    describe the same image.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    img += np.linspace(40, 200, size, dtype=np.float32)[None, :, None]

    yy, xx = np.mgrid[0:size, 0:size]
    for cell, amp in ((35, 40.0), (13, 25.0), (5, 15.0)):
        checker = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
        img += ((checker - 0.5) * 2.0 * amp)[..., None]

    for _ in range(6):
        x0, y0 = rng.integers(8, size - 72, size=2)
        w, h = rng.integers(24, 64, size=2)
        img[y0 : y0 + h, x0 : x0 + w] = rng.uniform(30, 220, size=3).astype(np.float32)

    img += np.array([6.0, 0.0, -6.0], np.float32)[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def sweep(steps: int) -> dict:
    scene = reference_scene()
    baseline = measure(scene)

    results = []
    for name in REGISTRY.names():
        deg = REGISTRY.get(name)
        for axis in deg.axes:
            points = []
            for value in np.linspace(axis.lo, axis.hi, steps):
                params = deg.Params(
                    **{a.field: (a.lo + a.hi) / 2.0 for a in deg.axes} | {axis.field: float(value)}
                )
                out = deg.apply(scene, params, seed=SEED)
                points.append(
                    {
                        "value": float(value),
                        "measured": {k: round(v, 6) for k, v in measure(out).items()},
                        "derived": {
                            k: {"value": round(v, 6), "unit": u}
                            for k, (v, u) in deg.derived(params).items()
                        },
                        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
                    }
                )
            results.append(
                {
                    "degradation": name,
                    "axis": axis.field,
                    "unit": axis.unit,
                    "range": [axis.lo, axis.hi],
                    "expect": list(axis.expect) if axis.expect else None,
                    "stochastic": deg.stochastic,
                    "points": points,
                }
            )

    return {
        "environment": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "seed": SEED,
        "scene_sha256": hashlib.sha256(scene.tobytes()).hexdigest(),
        "baseline_measurements": {k: round(v, 6) for k, v in baseline.items()},
        "axes": results,
        "command": "uv run python bench/degrade_response.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/degrade_response.json"))
    args = parser.parse_args()

    report = sweep(args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"OpenCV {report['environment']['opencv']} | seed {report['seed']}")
    print(f"scene sha256 {report['scene_sha256'][:16]}")
    print(f"{'axis':<44} {'unit':<14} endpoint response")
    print("-" * 92)
    for entry in report["axes"]:
        metric = entry["expect"][0] if entry["expect"] else "laplacian_var"
        lo = entry["points"][0]["measured"][metric]
        hi = entry["points"][-1]["measured"][metric]
        label = f"{entry['degradation']}.{entry['axis']}"
        print(f"{label:<44} {entry['unit']:<14} {metric}: {lo:.4g} -> {hi:.4g}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
