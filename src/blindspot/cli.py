"""Blindspot command line.

Three commands, and the important one takes three arguments:

    blindspot run   --dataset ./val/road100 --budget 0.40
    blindspot probe --degradation motion_blur --axis exposure_ms --value 12.6
    blindspot check --against report.json
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Find where your vision pipeline starts lying.",
    no_args_is_help=True,
)

DEFAULT_AXES = [
    "motion_blur.exposure_ms",
    "low_light.illuminance_lux",
    "fog.beta_per_m",
    "jpeg.quality",
]


def _load(dataset: pathlib.Path, model: pathlib.Path, frames: Optional[int]):
    from .runner.dataset import ValidationSet
    from .runner.yolox import COCO_CLASSES, YoloxPipeline

    validation_set = ValidationSet(dataset, COCO_CLASSES)
    loaded = validation_set.load(limit=frames)
    if not loaded:
        raise typer.BadParameter(f"no frames loaded from {dataset}")
    return validation_set, loaded, YoloxPipeline(model)


@app.command()
def run(
    dataset: pathlib.Path = typer.Option(..., help="Validation set directory"),
    budget: float = typer.Option(0.40, help="Hard spending limit for this run, USD"),
    strategy: str = typer.Option("active", help="active | grid"),
    model: pathlib.Path = typer.Option("models/yolox_s.onnx", help="Pipeline under test"),
    axes: list[str] = typer.Option(DEFAULT_AXES, "--axis", help="Axis to probe"),
    frames: Optional[int] = typer.Option(None, help="Limit frames (for a quick look)"),
    grid_steps: int = typer.Option(20, help="Grid resolution, and the precision target"),
    seed: int = typer.Option(20260906),
    out: pathlib.Path = typer.Option("report.json"),
    cost_per_probe: float = typer.Option(0.01, help="USD per probe, for the contract"),
):
    """Search for the pipeline's failure boundaries under a fixed budget."""
    import numpy as np

    from .boundary.coverage import axis_coverage, native_measurements, population_sweep
    from .boundary.criterion import Criterion
    from .boundary.locate import BoundaryStatus, locate_on_axis, scan_boundaries
    from .boundary.probe import baseline_map50, run_probe
    from .cost.contract import BudgetContract, BudgetExceeded
    from .degrade import REGISTRY

    if strategy not in {"active", "grid"}:
        raise typer.BadParameter("strategy must be 'active' or 'grid'")

    started = time.time()
    validation_set, loaded, pipeline = _load(dataset, model, frames)
    contract = BudgetContract(limit_usd=budget, cost_per_probe_usd=cost_per_probe)

    typer.echo(f"dataset  {validation_set.name}: {len(loaded)} frames, "
               f"{sum(len(f.truths) for f in loaded)} objects")
    typer.echo(f"pipeline {pipeline.name} via cv::dnn")
    typer.echo(f"budget   {budget:.2f} USD hard limit "
               f"({contract.probes_remaining} probes at {cost_per_probe:.3f} each)")

    base = baseline_map50(loaded, pipeline, validation_set)
    criterion = Criterion(baseline_map50=base["mAP50"])
    typer.echo(f"baseline mAP@50 {base['mAP50']:.4f}, "
               f"failure below {criterion.threshold:.4f}\n")

    findings = []
    coverage = []
    halted = None

    for axis_id in axes:
        degradation, _, axis_field = axis_id.partition(".")
        deg = REGISTRY.get(degradation)
        axis = deg.axis(axis_field)
        cell = (axis.hi - axis.lo) / (grid_steps - 1)

        probe_log = []

        def evaluate(value: float) -> bool:
            if not contract.can_afford(1):
                raise BudgetExceeded(contract.request(1).reason)
            contract.charge(1)
            result = run_probe(
                loaded, pipeline, validation_set, degradation, axis_field, value, seed
            )
            probe_log.append(result)
            return criterion.failed(result.map50)

        try:
            if strategy == "grid":
                samples = [
                    (float(axis.from_severity(s)), evaluate(float(axis.from_severity(s))))
                    for s in np.linspace(0.0, 1.0, grid_steps)
                ]
                transitions = [
                    tuple(sorted((axis.from_severity(a), axis.from_severity(b))))
                    for a, b in scan_boundaries(
                        [(axis.to_severity(v), f) for v, f in samples]
                    )
                ]
                status = "located" if len(transitions) == 1 else (
                    "not_monotone" if len(transitions) > 1 else "no_transition"
                )
                interval = transitions[0] if transitions else None
                probes_used = len(samples)
            else:
                located = locate_on_axis(
                    axis, evaluate, target_width=cell, verify_samples=5
                )
                status = located.status.value
                interval = (
                    (located.lower, located.upper)
                    if located.status is BoundaryStatus.LOCATED
                    else None
                )
                probes_used = located.probes_used
        except BudgetExceeded as exc:
            halted = {"axis": axis_id, "reason": str(exc)}
            typer.echo(f"  {axis_id}: HALTED -- {exc}")
            break

        failing = [p for p in probe_log if criterion.failed(p.map50)]
        witness = min(failing, key=lambda p: p.value) if failing else None

        finding = {
            "axis": axis_id,
            "unit": axis.unit,
            "status": status,
            "boundary": (
                {
                    "lower": interval[0],
                    "upper": interval[1],
                    "width": interval[1] - interval[0],
                    "unit": axis.unit,
                }
                if interval
                else None
            ),
            "probes_used": probes_used,
            "reproduce": witness.command if witness else None,
            "witness": witness.to_dict() if witness else None,
        }
        findings.append(finding)

        if interval:
            typer.echo(
                f"  {axis_id}: boundary {interval[0]:.4g} -> {interval[1]:.4g} "
                f"{axis.unit} ({probes_used} probes)"
            )
        else:
            typer.echo(f"  {axis_id}: {status} ({probes_used} probes)")

        # Coverage, from the probes already paid for.
        if axis.expect and probe_log:
            metric = axis.expect[0]
            ordered = sorted(probe_log, key=lambda p: p.value)
            coverage.append(
                axis_coverage(
                    axis_id=axis_id,
                    unit=axis.unit,
                    axis_range=(axis.lo, axis.hi),
                    metric=metric,
                    sweep_values=[p.value for p in ordered],
                    # Population median, not the first frame: sharpness and
                    # contrast are scene-dependent.
                    sweep_metric=population_sweep(
                        loaded, degradation, axis_field,
                        [p.value for p in ordered], metric, seed,
                    ),
                    native_metric_values=native_measurements(loaded, metric),
                ).to_dict()
            )

    report = {
        # Rule R1: what was never checked comes before what passed.
        "uncovered_regions": coverage,
        "findings": findings,
        "halted": halted,
        "strategy": strategy,
        "criterion": criterion.describe(),
        "baseline": base,
        "budget": contract.describe(),
        "budget_ledger": contract.ledger,
        "dataset": {
            "name": validation_set.name,
            "frames": len(loaded),
            "objects": sum(len(f.truths) for f in loaded),
            "source": validation_set.manifest["source"],
        },
        "pipeline": pipeline.describe(),
        "seed": seed,
        "wall_seconds": round(time.time() - started, 2),
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"\nspent {contract.spent_usd:.4f} of {budget:.2f} USD "
               f"({contract.probes_charged} probes)")
    if halted:
        typer.echo("run halted on the budget contract; partial results were kept")
    for entry in coverage:
        typer.echo(f"uncovered on {entry['axis']}: "
                   f"{100 - entry['coverage_percent']:.1f}% of the swept range")
    typer.echo(f"wrote {out}")


def _pairs(items: list[str], flag: str) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise typer.BadParameter(f"{flag} expects AXIS=VALUE, got {item!r}")
        try:
            out[name] = float(value)
        except ValueError:
            raise typer.BadParameter(f"{flag} value must be a number: {item!r}") from None
    return out


@app.command()
def probe(
    degradation: Optional[str] = typer.Option(None, help="Degradation name (single axis)"),
    axis: Optional[str] = typer.Option(None, help="Axis field to set (single axis)"),
    value: Optional[float] = typer.Option(None, help="Axis value, in the axis's physical unit"),
    set_: list[str] = typer.Option([], "--set", help="AXIS=VALUE, repeatable (composite)"),
    fix: list[str] = typer.Option([], "--fix", help="PARAM=VALUE pinned across kernels"),
    dataset: pathlib.Path = typer.Option("val/road100"),
    model: pathlib.Path = typer.Option("models/yolox_s.onnx"),
    frames: Optional[int] = typer.Option(None),
    seed: int = typer.Option(20260906),
):
    """Reproduce one condition. This is the command a report prints.

    Single axis:  --degradation motion_blur --axis exposure_ms --value 12.6
    Composite:    --set motion_blur.exposure_ms=12 --set low_light.illuminance_lux=18
                  --fix low_light.exposure_ms=12
    """
    single = degradation is not None or axis is not None or value is not None
    if set_ and single:
        raise typer.BadParameter("use either --set or --degradation/--axis/--value, not both")
    if not set_ and (degradation is None or axis is None or value is None):
        raise typer.BadParameter("give --set AXIS=VALUE, or all of --degradation --axis --value")
    axes = _pairs(set_, "--set")
    fixed = _pairs(fix, "--fix")

    validation_set, loaded, pipeline = _load(dataset, model, frames)
    if set_:
        from .boundary.probe import run_condition_probe
        from .degrade.compose import Condition

        result = run_condition_probe(
            loaded, pipeline, validation_set, Condition.from_axes(axes, fixed=fixed), seed
        )
        result.pop("captured", None)
        typer.echo(json.dumps(result, indent=2))
        return

    from .boundary.probe import run_probe

    result = run_probe(loaded, pipeline, validation_set, degradation, axis, value, seed)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def check(
    against: pathlib.Path = typer.Option(..., help="A previous report to compare with"),
    current: pathlib.Path = typer.Option("report.json", help="The report to check"),
):
    """CI gate: fail when a boundary has moved backwards.

    A boundary that has retreated means the pipeline now fails under milder
    conditions than it used to. That is a regression whether or not aggregate
    accuracy moved.
    """
    baseline = json.loads(against.read_text(encoding="utf-8"))
    latest = json.loads(current.read_text(encoding="utf-8"))

    previous = {
        f["axis"]: f["boundary"] for f in baseline["findings"] if f.get("boundary")
    }
    regressions = []
    for finding in latest["findings"]:
        old = previous.get(finding["axis"])
        new = finding.get("boundary")
        if not old or not new:
            continue
        # For axes where a larger value is more severe, a boundary that moved
        # *down* means failure now starts earlier.
        if new["lower"] < old["lower"]:
            regressions.append(
                f"{finding['axis']}: boundary moved {old['lower']:.4g} -> "
                f"{new['lower']:.4g} {new['unit']}"
            )

    if regressions:
        typer.echo("boundary regressions:")
        for line in regressions:
            typer.echo(f"  {line}")
        raise typer.Exit(code=1)

    typer.echo(f"no boundary regression across {len(previous)} axes")


if __name__ == "__main__":
    app()
