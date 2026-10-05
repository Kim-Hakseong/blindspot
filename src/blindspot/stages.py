"""Per-stage timing of one fixed probe batch, for comparing CPUs and builds.

The same 64 probes (16 per axis) run on every arm of the comparison -- x86,
Graviton with the stock OpenCV 5 wheel, Graviton with COOL -- and each frame
is timed in three stages: degradation (OpenCV), objective measurement
(OpenCV), and inference (cv::dnn). A speed-up is therefore attributed to a
stage rather than asserted for the whole pipeline. mAP is recorded per probe,
because an arm that is faster but scores differently is computing something
else -- and across CPU families the low decimals do differ (see the
cross-architecture record).
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import statistics
import time

import cv2
import numpy as np

AXES = ("motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m", "jpeg.quality")
SEED = 20260906


def fixture_probes() -> list[dict]:
    from .degrade import REGISTRY

    probes = []
    for axis_id in AXES:
        deg, _, field = axis_id.partition(".")
        a = REGISTRY.get(deg).axis(field)
        for k, v in enumerate(np.linspace(a.lo, a.hi, 16)):
            probes.append({"probe_id": f"{axis_id}#{k:02d}", "set": {axis_id: float(v)}})
    return probes


def _cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            key = line.split(":")[0].strip().lower()
            if key in ("model name", "cpu part", "hardware"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def fingerprint() -> dict:
    info = cv2.getBuildInformation()
    return {
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": cv2.getNumberOfCPUs(),
        "opencv_threads": cv2.getNumThreads(),
        "opencv_version": cv2.__version__,
        "cv2_module": getattr(cv2, "__file__", None),
        "build_info_sha256": hashlib.sha256(info.encode()).hexdigest(),
        "kleidicv": "kleidicv" in info.lower(),
        "python": platform.python_version(),
    }


def run_stage_bench(dataset, frames: int, probes: list[dict], label: str,
                    pipeline_name: str = "yolox_s") -> dict:
    from .degrade.compose import Condition, apply_condition
    from .measure import measure
    from .metrics import mean_ap50
    from .runner.dataset import ValidationSet
    from .runner.registry import load_pipeline
    from .runner.yolox import COCO_CLASSES

    vs = ValidationSet(dataset, COCO_CLASSES)
    loaded = vs.load(limit=frames)
    pipeline = load_pipeline(pipeline_name)
    for f in loaded[:2]:  # warm-up: dnn graph initialisation, excluded
        pipeline.predict(f.image, f.image_id)

    stage = {"degrade": [], "measure": [], "infer": []}
    results = []
    started = time.perf_counter()
    for probe in probes:
        cond = Condition.from_axes(probe["set"])
        preds, truths = [], []
        for f in loaded:
            t0 = time.perf_counter()
            img = apply_condition(f.image, cond, seed=SEED)
            t1 = time.perf_counter()
            measure(img)
            t2 = time.perf_counter()
            preds += vs.filter_predictions(pipeline.predict(img, f.image_id))
            t3 = time.perf_counter()
            stage["degrade"].append(t1 - t0)
            stage["measure"].append(t2 - t1)
            stage["infer"].append(t3 - t2)
            truths += f.truths
        results.append({"probe_id": probe["probe_id"], "map50": mean_ap50(preds, truths)["mAP50"]})
    wall = time.perf_counter() - started

    def summary(xs):
        ms = np.array(xs) * 1000.0
        return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
                "total_s": float(ms.sum() / 1000.0), "n": len(xs)}

    total = [(a + b + c) * 1000.0 for a, b, c in zip(stage["degrade"], stage["measure"], stage["infer"])]
    return {
        "label": label, "fingerprint": fingerprint(), "pipeline": pipeline_name,
        "frames_per_probe": len(loaded), "probes": len(probes), "seed": SEED,
        "wall_seconds": wall,
        "per_frame": {k: summary(v) for k, v in stage.items()},
        "per_frame_total_median_ms": statistics.median(total) if total else None,
        "results": results,
    }
