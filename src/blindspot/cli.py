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


def _load(dataset: pathlib.Path, pipeline: str, frames: Optional[int]):
    from .runner.dataset import ValidationSet
    from .runner.registry import PIPELINES, load_pipeline
    from .runner.yolox import COCO_CLASSES

    if pipeline not in PIPELINES:
        raise typer.BadParameter(f"unknown pipeline {pipeline!r}; known: {sorted(PIPELINES)}")
    validation_set = ValidationSet(dataset, COCO_CLASSES)
    loaded = validation_set.load(limit=frames)
    if not loaded:
        raise typer.BadParameter(f"no frames loaded from {dataset}")
    return validation_set, loaded, load_pipeline(pipeline)


@app.command()
def run(
    dataset: pathlib.Path = typer.Option(..., help="Validation set directory"),
    budget: float = typer.Option(0.40, help="Hard spending limit for this run, USD"),
    strategy: str = typer.Option("active", help="active | grid"),
    pipeline: str = typer.Option("yolox_s", help="Pipeline under test (see runner/registry.py)"),
    axes: list[str] = typer.Option(DEFAULT_AXES, "--axis", help="Axis to probe"),
    frames: Optional[int] = typer.Option(None, help="Limit frames (for a quick look)"),
    grid_steps: int = typer.Option(20, help="Grid resolution, and the precision target"),
    seed: int = typer.Option(20260906),
    out: pathlib.Path = typer.Option("report.json"),
    cost_per_probe: float = typer.Option(0.01, help="USD per probe, for the contract"),
    probes: Optional[int] = typer.Option(
        None, help="Contract in probes instead of USD: budget = probes x cost-per-probe"),
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

    if probes is not None:
        if probes < 1:
            raise typer.BadParameter("--probes must be at least 1")
        budget = probes * cost_per_probe
    started = time.time()
    validation_set, loaded, pipeline = _load(dataset, pipeline, frames)
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
    pipeline: str = typer.Option("yolox_s", help="Pipeline under test (see runner/registry.py)"),
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

    validation_set, loaded, pipeline = _load(dataset, pipeline, frames)
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
def worker(
    spec: str = typer.Option(..., help="Wave spec: local path or s3://bucket/key"),
    table: Optional[str] = typer.Option(None, help="DynamoDB probe table (cloud); else --ledger"),
    ledger: pathlib.Path = typer.Option("runs/ledger.jsonl", help="Local JSONL ledger"),
):
    """Run one wave of probes. This is the container's command on AWS Batch."""
    import os

    from .cloud.worker import DynamoSink, LocalSink, fetch_dataset, run_wave, select_probes

    if spec.startswith("s3://"):
        import boto3

        bucket, _, key = spec[len("s3://"):].partition("/")
        wave = json.loads(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
    else:
        wave = json.loads(pathlib.Path(spec).read_text(encoding="utf-8"))

    wave = select_probes(wave, os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    if str(wave["dataset"]).startswith("s3://"):
        wave["dataset_ref"] = wave["dataset"]
        wave["dataset"] = str(fetch_dataset(wave["dataset"], pathlib.Path("/tmp/datasets")))

    if table:
        import boto3

        sink = DynamoSink(boto3.resource("dynamodb").Table(table))
    else:
        sink = LocalSink(ledger)
    for record in run_wave(wave, sink):
        typer.echo(f"{record['probe_id']}: mAP@50 {record['map50']:.4f} "
                   f"({record['wall_seconds']}s on {record['arch']})")


@app.command("cloud-run")
def cloud_run(
    dataset: pathlib.Path = typer.Option(..., help="Local validation set to upload once"),
    budget: float = typer.Option(0.40, help="Hard spending limit for this run, USD"),
    arch: str = typer.Option("arm64", help="arm64 (Graviton) | x86"),
    axes: list[str] = typer.Option(DEFAULT_AXES, "--axis"),
    pipeline: str = typer.Option("yolox_s"),
    frames: Optional[int] = typer.Option(None),
    grid_steps: int = typer.Option(33, help="Precision target: axis range / (steps - 1)"),
    seed: int = typer.Option(20260906),
    cost_per_probe: float = typer.Option(0.002, help="USD per probe, for the contract"),
    stack: str = typer.Option("Blindspot"),
    profile: str = typer.Option("blindspot", help="AWS profile dedicated to this project"),
):
    """Start a run on AWS: upload the dataset, fix the contract, start the loop."""
    import hashlib
    import uuid

    import boto3

    from .degrade import REGISTRY

    if arch not in {"arm64", "x86"}:
        raise typer.BadParameter("arch must be arm64 or x86")
    session = boto3.Session(profile_name=profile)
    outputs = {o["OutputKey"]: o["OutputValue"] for o in session.client("cloudformation")
               .describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}
    bucket = outputs["BucketName"]
    s3 = session.client("s3")

    manifest = (dataset / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()[:12]
    prefix = f"datasets/{dataset.name}-{digest}"
    for path in [dataset / "manifest.json", *sorted((dataset / "images").glob("*"))]:
        key = f"{prefix}/{path.relative_to(dataset).as_posix()}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except s3.exceptions.ClientError:
            s3.upload_file(str(path), bucket, key)

    run_axes = []
    for axis_id in axes:
        deg, _, field = axis_id.partition(".")
        a = REGISTRY.get(deg).axis(field)
        run_axes.append({"axis": axis_id, "unit": a.unit, "lo": a.lo, "hi": a.hi,
                         "severe_end": a.severe_end,
                         "target_width": (a.hi - a.lo) / (grid_steps - 1)})
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    definition = {"run_id": run_id, "seed": seed, "verify_samples": 5, "budget_usd": budget,
                  "cost_per_probe_usd": cost_per_probe, "dataset": f"s3://{bucket}/{prefix}",
                  "pipeline": pipeline, "frames": frames, "axes": run_axes}
    session.resource("dynamodb").Table("bs-runs").put_item(Item={
        "run_id": run_id, "status": "RUNNING", "arch": arch, "round": 0,
        "definition": json.dumps(definition), "created": int(time.time())})
    execution = session.client("stepfunctions").start_execution(
        stateMachineArn=outputs["StateMachineArn"], name=run_id,
        input=json.dumps({"run_id": run_id, "arch": arch}))
    typer.echo(f"run {run_id} started on {arch}; budget {budget:.2f} USD fixed")
    typer.echo(execution["executionArn"])


@app.command()
def approve(
    run_id: str = typer.Option(..., help="Run halted in AWAITING_APPROVAL"),
    additional_usd: float = typer.Option(..., help="Amount to add to the contract"),
    approver: str = typer.Option(..., help="Who is approving"),
    reason: str = typer.Option(..., help="Why"),
    profile: str = typer.Option("blindspot"),
    stack: str = typer.Option("Blindspot"),
):
    """Human approval: continue a halted run under a new, larger contract."""
    import boto3

    from .cloud.approve import approve as do_approve

    session = boto3.Session(profile_name=profile)
    ddb = session.resource("dynamodb")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in session.client("cloudformation")
               .describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}
    sfn = session.client("stepfunctions")

    def start(run_id, arch, contract_version):
        sfn.start_execution(stateMachineArn=outputs["StateMachineArn"],
                            name=f"{run_id}-v{contract_version}",
                            input=json.dumps({"run_id": run_id, "arch": arch}))

    out = do_approve(ddb.Table("bs-runs"), ddb.Table("bs-decisions"), start,
                     run_id, additional_usd, approver, reason)
    typer.echo(f"run {run_id} resumed under contract v{out['contract_version']}: "
               f"{out['budget_usd']:.2f} USD")


@app.command("bench-stages")
def bench_stages(
    label: str = typer.Option(..., help="Arm name, e.g. fargate-arm64 / fargate-x86 / cool-c8g"),
    dataset: str = typer.Option("val/road100", help="Local path or s3://bucket/datasets/<name>"),
    frames: int = typer.Option(10),
    out: str = typer.Option("bench/out/cool", help="Local directory or s3://bucket/prefix"),
):
    """Time each stage of the fixed 64-probe batch (CPU / build comparison)."""
    from .stages import fixture_probes, run_stage_bench

    if dataset.startswith("s3://"):
        from .cloud.worker import fetch_dataset

        dataset = str(fetch_dataset(dataset, pathlib.Path("/tmp/datasets")))
    report = run_stage_bench(dataset, frames, fixture_probes(), label)
    report["command"] = f"blindspot bench-stages --label {label} --frames {frames}"
    body = json.dumps(report, indent=1) + "\n"
    name = f"{label}.json"
    if out.startswith("s3://"):
        import boto3

        bucket, _, prefix = out[len("s3://"):].partition("/")
        boto3.client("s3").put_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/{name}",
                                      Body=body, ContentType="application/json")
    else:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out) / name).write_text(body)
    fp = report["fingerprint"]
    typer.echo(f"{label}: {fp['cpu_model']} ({fp['machine']}), OpenCV {fp['opencv_version']}, "
               f"kleidicv={fp['kleidicv']}")
    for stage, s in report["per_frame"].items():
        typer.echo(f"  {stage:<8} median {s['median_ms']:.2f} ms  p95 {s['p95_ms']:.2f} ms")


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
    from .degrade import REGISTRY

    baseline = json.loads(against.read_text(encoding="utf-8"))
    latest = json.loads(current.read_text(encoding="utf-8"))

    def pass_edge_severity(axis_id: str, boundary: dict) -> float:
        """How harsh the last passing condition is, in [0, 1].

        Comparing in severity rather than raw value is what makes this right on
        axes whose severe end is low (lux, JPEG quality), where failing
        *sooner* means the boundary moved up.
        """
        deg, _, field = axis_id.partition(".")
        axis = REGISTRY.get(deg).axis(field)
        edge = boundary["lower"] if axis.severe_end == "hi" else boundary["upper"]
        return axis.to_severity(edge)

    previous = {f["axis"]: f for f in baseline["findings"]}
    regressions = []
    for finding in latest["findings"]:
        old = previous.get(finding["axis"])
        new = finding.get("boundary")
        if old is None or not new:
            continue
        if not old.get("boundary"):
            if old.get("status") == "passes_throughout":
                regressions.append(f"{finding['axis']}: a failure boundary appeared where the "
                                   "whole range used to pass")
            continue
        before = pass_edge_severity(finding["axis"], old["boundary"])
        after = pass_edge_severity(finding["axis"], new)
        if after < before - 1e-12:
            regressions.append(
                f"{finding['axis']}: now fails from a milder condition "
                f"({old['boundary']['lower']:.4g}-{old['boundary']['upper']:.4g} -> "
                f"{new['lower']:.4g}-{new['upper']:.4g} {new.get('unit', '')})"
            )

    if regressions:
        typer.echo("boundary regressions:")
        for line in regressions:
            typer.echo(f"  {line}")
        raise typer.Exit(code=1)

    typer.echo(f"no boundary regression across {len(previous)} axes")


if __name__ == "__main__":
    app()
