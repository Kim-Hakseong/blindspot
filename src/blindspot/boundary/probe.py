"""One probe: evaluate a pipeline under one condition.

A probe is the unit of work and the unit of cost. It is
`(validation set, degradation, physical parameters, seed) -> metrics`, and it
carries everything needed to reproduce itself, because a boundary is only
useful if the frame that broke can be put back on screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from ..degrade import REGISTRY
from ..measure import measure
from ..metrics import mean_ap50


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one condition, with its reproduction recipe."""

    degradation: str
    axis: str
    value: float
    unit: str
    seed: int
    map50: float
    counts: dict[str, float]
    derived: dict[str, dict[str, float | str]]
    measured: dict[str, float]
    n_frames: int
    wall_seconds: float
    params: dict[str, float] = field(default_factory=dict)

    @property
    def command(self) -> str:
        """The command that regenerates this exact condition."""
        args = " ".join(f"--{k} {v:g}" for k, v in sorted(self.params.items()))
        return (
            f"uv run blindspot probe --degradation {self.degradation} "
            f"{args} --seed {self.seed}"
        )

    def to_dict(self) -> dict:
        return {
            "degradation": self.degradation,
            "axis": self.axis,
            "value": self.value,
            "unit": self.unit,
            "seed": self.seed,
            "map50": self.map50,
            "counts": self.counts,
            "derived": self.derived,
            "measured_degradation": self.measured,
            "n_frames": self.n_frames,
            "wall_seconds": round(self.wall_seconds, 3),
            "params": self.params,
            "command": self.command,
        }


def run_probe(
    frames: Sequence,
    pipeline,
    validation_set,
    degradation: str,
    axis: str,
    value: float,
    seed: int,
) -> ProbeResult:
    """Degrade every frame at one condition, run the pipeline, score it."""
    started = time.perf_counter()
    deg = REGISTRY.get(degradation)
    axis_spec = deg.axis(axis)

    values = {a.field: (a.lo + a.hi) / 2.0 for a in deg.axes}
    values[axis] = float(value)
    params = deg.Params(**values)

    predictions = []
    truths = []
    # Measure the objective degradation on the first frame only: it is a
    # property of the condition, and doing it per frame would dominate cost.
    measured: dict[str, float] = {}

    for index, frame in enumerate(frames):
        degraded = deg.apply(frame.image, params, seed=seed)
        if index == 0:
            measured = {k: round(v, 6) for k, v in measure(degraded).items()}
        predictions.extend(
            validation_set.filter_predictions(pipeline.predict(degraded, frame.image_id))
        )
        truths.extend(frame.truths)

    scored = mean_ap50(predictions, truths)
    counts = {k: v for k, v in scored.items() if k != "mAP50"}

    return ProbeResult(
        degradation=degradation,
        axis=axis,
        value=float(value),
        unit=axis_spec.unit,
        seed=seed,
        map50=scored["mAP50"],
        counts=counts,
        derived={
            k: {"value": round(v, 6), "unit": u} for k, (v, u) in deg.derived(params).items()
        },
        measured=measured,
        n_frames=len(frames),
        wall_seconds=time.perf_counter() - started,
        params={k: float(v) for k, v in params.model_dump().items()},
    )


def baseline_map50(frames: Sequence, pipeline, validation_set) -> dict[str, float]:
    """Score the pipeline with no degradation applied."""
    predictions = []
    truths = []
    for frame in frames:
        predictions.extend(
            validation_set.filter_predictions(pipeline.predict(frame.image, frame.image_id))
        )
        truths.extend(frame.truths)
    return mean_ap50(predictions, truths)
