"""The probe worker: run one wave of probes and write the ledger.

A wave is a JSON spec -- run id, dataset, pipeline, seed and a list of
conditions -- and the worker scores each condition on the validation set and
writes one record per probe (rule C2). It reports mAP@50 and counts only. It
does not say pass or fail: that needs the run's undegraded baseline and the
criterion, which the planner holds.

The same code runs on a laptop, inside the local Docker image, and as an AWS
Batch job on Graviton or x86. Only the sink and the dataset location differ.
"""

from __future__ import annotations

import json
import pathlib
import platform
import time

import cv2

from ..boundary.probe import run_condition_probe
from ..degrade.compose import Condition

#: Every ledger record carries these (rule C2).
REQUIRED_FIELDS = {
    "run_id", "probe_id", "condition", "seed", "map50", "counts", "n_frames",
    "wall_seconds", "pipeline", "dataset", "arch", "opencv", "command",
}


class LocalSink:
    """Append-only JSONL ledger on local disk."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _command(probe: dict, seed: int, frames: int | None, pipeline: str, dataset: str) -> str:
    parts = ["uv run blindspot probe"]
    parts += [f"--set {k}={float(v)!r}" for k, v in probe.get("set", {}).items()]
    parts += [f"--fix {k}={float(v)!r}" for k, v in probe.get("fix", {}).items()]
    parts += [f"--dataset {dataset}", f"--seed {seed}"]
    if pipeline != "yolox_s":
        parts.append(f"--pipeline {pipeline}")
    if frames:
        parts.append(f"--frames {frames}")
    return " ".join(parts)


def run_wave(spec: dict, sink) -> list[dict]:
    from ..runner.dataset import ValidationSet
    from ..runner.registry import load_pipeline
    from ..runner.yolox import COCO_CLASSES

    # Validate every condition before spending anything.
    conditions = [
        Condition.from_axes(p.get("set", {}), fixed=p.get("fix", {})) for p in spec["probes"]
    ]
    vs = ValidationSet(spec["dataset"], COCO_CLASSES)
    frames = vs.load(limit=spec.get("frames"))
    pipeline = load_pipeline(spec["pipeline"])

    results = []
    for probe, condition in zip(spec["probes"], conditions):
        started = time.perf_counter()
        scored = run_condition_probe(frames, pipeline, vs, condition, spec["seed"])
        record = {
            "run_id": spec["run_id"],
            "probe_id": probe["probe_id"],
            "condition": scored["condition"],
            "seed": spec["seed"],
            "map50": scored["map50"],
            "counts": scored["counts"],
            "n_frames": scored["n_frames"],
            "wall_seconds": round(time.perf_counter() - started, 3),
            "pipeline": spec["pipeline"],
            "dataset": vs.name,
            "arch": platform.machine(),
            "opencv": cv2.__version__,
            "command": _command(probe, spec["seed"], spec.get("frames"),
                                spec["pipeline"], spec.get("dataset_ref", f"val/{vs.name}")),
        }
        sink.write(record)
        results.append(record)
    return results
