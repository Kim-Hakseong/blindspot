"""The 2-D Blindspot Map: exposure time x illuminance, one camera.

A real camera has one exposure time, and it sets two things at once: how far
the image smears during capture (motion blur) and how much light the sensor
collects (shot noise). So this map couples them -- `low_light.exposure_ms` is
pinned to the same value as `motion_blur.exposure_ms` -- rather than letting a
probe combine a 40 ms blur with a 10 ms sensor integration, a camera that
cannot exist.

Grid: exposure linear, illuminance geometric (light is perceived and behaves
multiplicatively; a linear lux grid would spend almost every probe in bright
conditions). Both have 2^k+1 points so coarser grids nest inside finer ones,
and the viewer's frame images come from the nested 9x9 subset.

The camera is modelled at fixed gain (no auto-exposure), with angular velocity
at the axis midpoint. Those are stated in the output, not hidden.

Every cell is exhaustively probed on the full validation set. This file is also
the ground truth that 2-D boundary sampling (W3-3) is measured against.

    uv run python bench/hook_grid.py --steps 17
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
from blindspot.boundary.probe import baseline_map50, run_condition_probe  # noqa: E402
from blindspot.boundary.store import ProbeStore  # noqa: E402
from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.degrade.compose import Condition  # noqa: E402

X_AXIS = "motion_blur.exposure_ms"
Y_AXIS = "low_light.illuminance_lux"
EXPOSURE_RANGE = (1.0, 40.0)   # ms; the sensor model needs a nonzero exposure
LUX_RANGE = (400.0, 0.5)       # benign -> severe


def grid_values(steps: int) -> tuple[list[float], list[float]]:
    xs = [float(v) for v in np.linspace(*EXPOSURE_RANGE, steps)]
    ys = [float(v) for v in np.geomspace(*LUX_RANGE, steps)]
    return xs, ys


def condition(exposure_ms: float, lux: float) -> Condition:
    return Condition.from_axes(
        {X_AXIS: exposure_ms, Y_AXIS: lux},
        fixed={"low_light.exposure_ms": exposure_ms},
    )


def main() -> int:
    from blindspot.privacy import FaceBlurrer
    from blindspot.runner.dataset import ValidationSet
    from blindspot.runner.yolox import COCO_CLASSES, YoloxPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("val/road100"))
    parser.add_argument("--steps", type=int, default=17)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--image-steps", type=int, default=9,
                        help="nested sub-grid for which the demo frame is saved")
    parser.add_argument("--demo-frame", default=None,
                        help="image id to capture; default is the motion-blur evidence frame")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/hook_grid.json"))
    parser.add_argument("--cells-dir", type=pathlib.Path, default=pathlib.Path("bench/out/hook_cells"))
    parser.add_argument("--probe-cache", type=pathlib.Path, default=pathlib.Path(".cache/probes2d.json"))
    args = parser.parse_args()

    if (args.steps - 1) % (args.image_steps - 1):
        sys.exit("image-steps must nest inside steps (both 2^k+1)")

    vs = ValidationSet(args.dataset, COCO_CLASSES)
    frames = vs.load(limit=args.frames)
    pipeline = YoloxPipeline()
    store = ProbeStore(args.probe_cache)
    blur = FaceBlurrer()

    demo = args.demo_frame
    if demo is None:
        frames_index = json.loads(pathlib.Path("bench/out/frames/frames.json").read_text())
        demo = next(e for e in frames_index if e["axis"] == X_AXIS)["frame"]["image_id"]
    demo_record = next(r for r in vs.manifest["images"] if r["image_id"] == demo)
    if demo not in {f.image_id for f in frames}:
        sys.exit(f"demo frame {demo} not in the loaded frames")

    base = baseline_map50(frames, pipeline, vs)
    criterion = Criterion(baseline_map50=base["mAP50"])
    xs, ys = grid_values(args.steps)
    stride = (args.steps - 1) // (args.image_steps - 1)
    args.cells_dir.mkdir(parents=True, exist_ok=True)

    print(f"{vs.name}: {len(frames)} frames; baseline mAP@50 {base['mAP50']:.4f}; "
          f"threshold {criterion.threshold:.4f}; {args.steps}x{args.steps} cells")

    started = time.perf_counter()
    cells = []
    for j, lux in enumerate(ys):
        row = []
        for i, exposure in enumerate(xs):
            cond = condition(exposure, lux)
            want_image = (i % stride == 0) and (j % stride == 0)
            key = ProbeStore.key(vs.name, len(frames), pipeline.name, cond.describe(), args.seed)
            cached = store.get(key)
            image_path = args.cells_dir / f"x{i:02d}_y{j:02d}.jpg"
            if cached is None or (want_image and not image_path.is_file()):
                result = run_condition_probe(frames, pipeline, vs, cond, args.seed,
                                             capture=demo if want_image else None)
                record = {k: v for k, v in result.items() if k != "captured"}
                store.put(key, record)
                if want_image:
                    cap = result["captured"]
                    cv2.imwrite(str(image_path), blur(cap["image"]),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    (image_path.with_suffix(".json")).write_text(json.dumps({
                        "predictions": [
                            {"box": [round(v, 1) for v in m.detection.box],
                             "label": COCO_CLASSES[m.detection.label],
                             "score": round(m.detection.score, 4),
                             "matched": m.matched}
                            for m in cap["matches"]
                        ],
                        "truths": [{"box": [round(v, 1) for v in t.box],
                                    "label": COCO_CLASSES[t.label]} for t in cap["truths"]],
                    }))
                cached = record
            psf = REGISTRY.get("motion_blur").derived(
                cond.params(cond.steps[0]))["psf_length_px"][0]
            row.append({
                "i": i, "j": j,
                "exposure_ms": exposure, "illuminance_lux": lux,
                "psf_length_px": round(psf, 4),
                "map50": round(cached["map50"], 6),
                "failed": criterion.failed(cached["map50"]),
                "counts": cached["counts"],
                "image": image_path.name if want_image else None,
            })
            print(f"  x={exposure:6.2f}ms y={lux:8.3g}lux  mAP {cached['map50']:.3f} "
                  f"{'FAIL' if row[-1]['failed'] else 'pass'}", flush=True)
        cells.append(row)

    mb = REGISTRY.get("motion_blur")
    report = {
        "x": {"axis": X_AXIS, "unit": "ms", "values": xs, "spacing": "linear"},
        "y": {"axis": Y_AXIS, "unit": "lux", "values": ys, "spacing": "geometric"},
        "coupling": "low_light.exposure_ms is pinned to motion_blur.exposure_ms (one camera, one exposure)",
        "camera_model": {
            "angular_velocity_deg_s": (mb.axis("angular_velocity_deg_s").lo
                                       + mb.axis("angular_velocity_deg_s").hi) / 2.0,
            "focal_length_px": mb.Params.model_fields["focal_length_px"].default,
            "iso_gain": REGISTRY.get("low_light").Params.model_fields["iso_gain"].default,
            "auto_exposure": False,
        },
        "steps": args.steps,
        "image_steps": args.image_steps,
        "seed": args.seed,
        "dataset": {"name": vs.name, "frames": len(frames),
                    "objects": sum(len(f.truths) for f in frames)},
        "pipeline": pipeline.describe(),
        "criterion": criterion.describe(),
        "baseline": base,
        "demo_frame": {
            "image_id": demo,
            "attribution": {"source": demo_record["flickr_url"],
                            "license": demo_record["license_name"],
                            "license_url": demo_record["license_url"]},
            "selection": "motion-blur evidence frame chosen by bench/failure_frames.py::pick_frame",
            "faces": "blurred at render time with YuNet, after scoring",
        },
        "n_failed": sum(c["failed"] for r in cells for c in r),
        "n_cells": args.steps * args.steps,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "environment": {"opencv": cv2.__version__, "python": platform.python_version(),
                        "machine": platform.machine()},
        "command": f"uv run python bench/hook_grid.py --steps {args.steps}",
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{report['n_failed']}/{report['n_cells']} cells fail; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
