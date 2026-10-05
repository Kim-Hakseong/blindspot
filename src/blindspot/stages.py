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
from collections import Counter

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
        "numpy_version": np.__version__,
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


def compare_arms(arms: dict[str, list[dict]], usd_per_task_hour: dict[str, float]) -> dict:
    """Compare stage-bench reports from two CPU arms, each run one or more times.

    Speed is the median over repeats of each run's per-frame median. Cost is
    that time priced at the arm's task-hour rate, so the comparison is per unit
    of work rather than per wall-clock hour. Agreement compares per-probe mAP
    between the first run of each arm.
    """
    out: dict = {"arms": {}}
    for arch, runs in arms.items():
        ms = statistics.median(r["per_frame_total_median_ms"] for r in runs)
        maps = [[p["map50"] for p in r["results"]] for r in runs]
        out["arms"][arch] = {
            "cpu_model": runs[0]["fingerprint"]["cpu_model"],
            # Fargate does not pin a CPU generation; an arm may span several.
            "cpu_models": dict(Counter(r["fingerprint"]["cpu_model"] for r in runs)),
            "runs": [{"cpu_model": r["fingerprint"]["cpu_model"],
                      "per_frame_total_median_ms": r["per_frame_total_median_ms"]} for r in runs],
            "kleidicv": runs[0]["fingerprint"].get("kleidicv"),
            "opencv_version": runs[0]["fingerprint"]["opencv_version"],
            "repeats": len(runs),
            "per_frame_total_median_ms": ms,
            "per_stage_median_ms": {s: statistics.median(r["per_frame"][s]["median_ms"] for r in runs)
                                    for s in runs[0]["per_frame"]},
            "usd_per_task_hour": usd_per_task_hour[arch],
            "usd_per_1000_frames": 1000 * ms / 3.6e6 * usd_per_task_hour[arch],
            "repeats_bit_identical": all(m == maps[0] for m in maps),
        }
    if {"arm64", "x86"} <= set(arms):
        a, x = out["arms"]["arm64"], out["arms"]["x86"]
        out["speedup_arm64_over_x86"] = x["per_frame_total_median_ms"] / a["per_frame_total_median_ms"]
        out["cost_ratio_x86_over_arm64"] = x["usd_per_1000_frames"] / a["usd_per_1000_frames"]
        pa = [p["map50"] for p in arms["arm64"][0]["results"]]
        px = [p["map50"] for p in arms["x86"][0]["results"]]
        diffs = [abs(i - j) for i, j in zip(pa, px)]
        out["map50_identical_probes"] = sum(d == 0.0 for d in diffs)
        out["map50_probes_compared"] = len(diffs)
        out["map50_max_abs_difference"] = max(diffs)
    return out


def _map_agreement(a: list[dict], b: list[dict]) -> dict:
    diffs = [abs(p["map50"] - q["map50"]) for p, q in zip(a[0]["results"], b[0]["results"])]
    return {"map50_identical_probes": sum(d == 0.0 for d in diffs),
            "map50_probes_compared": len(diffs), "map50_max_abs_difference": max(diffs)}


def compare_three_way(arms: dict[str, list[dict]], usd_per_hour: dict[str, float]) -> dict:
    """x86 stock / Graviton stock / Graviton COOL, with each effect isolated.

    The chip effect compares two arms that differ only in CPU (same stock
    OpenCV wheel); the COOL effect compares two arms on the same instance that
    differ only in the OpenCV build. A speedup above 1 means the candidate is
    faster; a cost ratio below 1 means it is cheaper per frame.
    """
    out = compare_arms(arms, usd_per_hour)
    for name, base, cand in (("chip_effect", "x86_stock", "graviton_stock"),
                             ("cool_effect", "graviton_stock", "graviton_cool")):
        b, c = out["arms"][base], out["arms"][cand]
        out[name] = {
            "baseline": base, "candidate": cand,
            "speedup": b["per_frame_total_median_ms"] / c["per_frame_total_median_ms"],
            "stage_speedup": {s: b["per_stage_median_ms"][s] / c["per_stage_median_ms"][s]
                              for s in b["per_stage_median_ms"]},
            "cost_ratio": c["usd_per_1000_frames"] / b["usd_per_1000_frames"],
            **_map_agreement(arms[base], arms[cand]),
        }
    return out


#: Fingerprint fields derived from a build's own files or build information.
BUILD_DERIVED = ("cv2_module", "build_info_sha256", "kleidicv")


def public_fingerprint(fp: dict, licensed_build: bool) -> dict:
    """What may be published about the OpenCV build a report ran on.

    For a licensed build (COOL, whose materials its licence treats as
    confidential) only the version string and our own measurements of the
    machine are kept; anything read from the build itself is dropped.
    """
    if not licensed_build:
        return fp
    return {k: v for k, v in fp.items() if k not in BUILD_DERIVED}
