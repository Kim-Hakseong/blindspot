"""The planner Lambda adapter, against in-memory fakes of DynamoDB and S3."""

from __future__ import annotations

import json

from blindspot.cloud import plan_handler
from blindspot.degrade import REGISTRY


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[(Item["run_id"], Item.get("probe_id") or Item.get("decision_id") or "")] = Item

    def get_item(self, Key):
        item = self.items.get((Key["run_id"], ""))
        return {"Item": item} if item else {}

    def query_run(self, run_id):
        return [v for (r, _), v in self.items.items() if r == run_id]


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body


def run_item(budget=0.40):
    a = REGISTRY.get("motion_blur").axis("exposure_ms")
    return {"run_id": "r1", "status": "RUNNING", "arch": "arm64", "definition": json.dumps({
        "run_id": "r1", "seed": 7, "verify_samples": 5, "budget_usd": budget,
        "cost_per_probe_usd": 0.01, "dataset": "s3://b/datasets/road100", "pipeline": "yolox_s",
        "frames": 100,
        "axes": [{"axis": "motion_blur.exposure_ms", "unit": a.unit, "lo": a.lo, "hi": a.hi,
                  "severe_end": a.severe_end, "target_width": (a.hi - a.lo) / 32}],
    })}


def wire(monkeypatch, budget=0.40):
    runs, probes, decisions, s3 = FakeTable(), FakeTable(), FakeTable(), FakeS3()
    runs.put_item(run_item(budget))
    monkeypatch.setattr(plan_handler, "_clients", lambda: (runs, probes, decisions, s3, "bucket"))
    return runs, probes, decisions, s3


def test_plan_writes_a_wave_spec_and_returns_its_size(monkeypatch):
    runs, probes, decisions, s3 = wire(monkeypatch)
    out = plan_handler.handler({"mode": "plan", "run_id": "r1", "arch": "arm64"}, None)
    assert out["action"] == "probe" and out["size"] == 1
    (bucket, key), body = next(iter(s3.objects.items()))
    assert out["spec"] == f"s3://{bucket}/{key}"
    spec = json.loads(body)
    assert spec["probes"][0]["probe_id"] == "baseline"
    assert spec["dataset"] == "s3://b/datasets/road100"


def test_every_plan_is_recorded_as_a_decision(monkeypatch):
    runs, probes, decisions, s3 = wire(monkeypatch)
    plan_handler.handler({"mode": "plan", "run_id": "r1", "arch": "arm64"}, None)
    recorded = decisions.query_run("r1")
    assert len(recorded) == 1
    d = recorded[0]
    assert d["tool"] == "scheduler.plan_round" and d["accepted_by_scheduler"] is True


def test_halt_marks_the_run_awaiting_approval(monkeypatch):
    runs, probes, decisions, s3 = wire(monkeypatch, budget=0.01)
    probes.put_item({"run_id": "r1", "probe_id": "baseline", "set": "{}", "map50": "0.6"})
    out = plan_handler.handler({"mode": "plan", "run_id": "r1", "arch": "arm64"}, None)
    assert out["action"] == "halted"
    plan_handler.handler({"mode": "halt", "run_id": "r1", "arch": "arm64"}, None)
    assert runs.items[("r1", "")]["status"] == "AWAITING_APPROVAL"


def test_each_plan_emits_cloudwatch_metrics(monkeypatch, capsys):
    """W4-6: Embedded Metric Format on stdout -- CloudWatch turns it into
    metrics without the function needing PutMetricData permission."""
    wire(monkeypatch)
    plan_handler.handler({"mode": "plan", "run_id": "r1", "arch": "arm64"}, None)
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    emf = next(l for l in lines if "_aws" in l)
    directive = emf["_aws"]["CloudWatchMetrics"][0]
    assert directive["Namespace"] == "Blindspot"
    names = {m["Name"] for m in directive["Metrics"]}
    assert {"ProbesCompleted", "SpentUSD", "AxesLocated", "WaveSize"} <= names
    assert emf["Arch"] == "arm64" and emf["ProbesCompleted"] == 0
