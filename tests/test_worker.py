"""The probe worker (W4-1 / W4-7).

One entrypoint runs a wave of probes and writes one ledger record per probe
(rule C2), locally to JSONL or in the cloud to DynamoDB. The local path is
tested end to end on two frames; the cloud sink is tested for its record shape
only, without AWS.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from blindspot.cloud.worker import REQUIRED_FIELDS, LocalSink, run_wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_assets = pytest.mark.skipif(
    not (ROOT / "models" / "yolox_s.onnx").is_file()
    or not (ROOT / "val" / "road100" / "manifest.json").is_file(),
    reason="model or dataset not fetched",
)


def spec(**over):
    s = {
        "run_id": "test-run",
        "dataset": str(ROOT / "val" / "road100"),
        "pipeline": "yolox_s",
        "seed": 7,
        "frames": 2,
        "probes": [
            {"probe_id": "baseline", "set": {}},
            {"probe_id": "p1", "set": {"motion_blur.exposure_ms": 20.0}},
            {"probe_id": "p2", "set": {"motion_blur.exposure_ms": 20.0,
                                       "low_light.illuminance_lux": 30.0},
             "fix": {"low_light.exposure_ms": 20.0}},
        ],
    }
    s.update(over)
    return s


@needs_assets
def test_one_ledger_record_per_probe(tmp_path):
    sink = LocalSink(tmp_path / "ledger.jsonl")
    results = run_wave(spec(), sink)
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == len(results) == 3
    for line in lines:
        record = json.loads(line)
        assert REQUIRED_FIELDS <= set(record), REQUIRED_FIELDS - set(record)


@needs_assets
def test_records_carry_a_runnable_reproduce_command(tmp_path):
    results = run_wave(spec(), LocalSink(tmp_path / "l.jsonl"))
    p2 = next(r for r in results if r["probe_id"] == "p2")
    assert "--set motion_blur.exposure_ms=20.0" in p2["command"]
    assert "--fix low_light.exposure_ms=20.0" in p2["command"]
    assert "--frames 2" in p2["command"]


@needs_assets
def test_wave_is_deterministic(tmp_path):
    a = run_wave(spec(), LocalSink(tmp_path / "a.jsonl"))
    b = run_wave(spec(), LocalSink(tmp_path / "b.jsonl"))
    assert [r["map50"] for r in a] == [r["map50"] for r in b]


@needs_assets
def test_worker_does_not_judge(tmp_path):
    """Pass/fail needs the run's baseline; the planner applies it, not the worker."""
    results = run_wave(spec(), LocalSink(tmp_path / "l.jsonl"))
    assert all("failed" not in r for r in results)


def test_unknown_axis_fails_before_any_work(tmp_path):
    bad = spec(probes=[{"probe_id": "x", "set": {"motion_blur.intensity": 1.0}}])
    with pytest.raises(KeyError):
        run_wave(bad, LocalSink(tmp_path / "l.jsonl"))


@needs_assets
def test_ledger_command_reruns_to_the_same_number(tmp_path):
    """R2 for the cloud path: a worker's record must be reproducible from the
    command stored in it."""
    import shlex

    from typer.testing import CliRunner

    from blindspot.cli import app

    record = run_wave(spec(pipeline="nanodet_plus_m", dataset_ref=str(ROOT / "val" / "road100")),
                      LocalSink(tmp_path / "l.jsonl"))[2]
    argv = shlex.split(record["command"])[4:]  # drop "uv run blindspot probe"
    out = CliRunner().invoke(app, ["probe", *argv])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output[out.output.index("{"):])["map50"] == record["map50"]


def test_array_child_runs_only_its_own_probe():
    from blindspot.cloud.worker import select_probes

    s = spec()
    assert select_probes(s, None)["probes"] == s["probes"]
    assert select_probes(s, "2")["probes"] == [s["probes"][2]]


def test_dynamo_sink_stores_decimals_and_serialised_condition():
    from decimal import Decimal

    from blindspot.cloud.worker import DynamoSink

    class T:
        item = None

        def put_item(self, Item):
            T.item = Item

    DynamoSink(T()).write({"run_id": "r", "probe_id": "p", "set": {"a.b": 1.5}, "map50": 0.25})
    assert T.item["map50"] == Decimal("0.25")
    assert T.item["set"] == '{"a.b": 1.5}'
