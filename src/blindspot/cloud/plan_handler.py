"""Lambda adapter around `plan_round`: read the ledger, write the next wave.

Modes, as the state machine calls them:
  plan      -> read run + probe ledger, plan the next round, write the wave spec
               to S3, record the decision in bs-decisions, return its size
  finalize  -> write the envelope (findings) to S3, mark the run DONE
  halt      -> mark the run AWAITING_APPROVAL (budget contract reached)

All reasoning lives in the pure `plan_round`; this module only moves data.
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal

from .plan import plan_round


def _clients():  # pragma: no cover - replaced in tests
    import boto3

    ddb = boto3.resource("dynamodb")
    return (ddb.Table(os.environ["RUNS_TABLE"]), ddb.Table(os.environ["PROBES_TABLE"]),
            ddb.Table(os.environ["DECISIONS_TABLE"]), boto3.client("s3"),
            os.environ["BLINDSPOT_BUCKET"])


def _ledger(probes_table, run_id: str) -> list[dict]:
    if hasattr(probes_table, "query_run"):  # test fake
        items = probes_table.query_run(run_id)
    else:  # pragma: no cover
        from boto3.dynamodb.conditions import Key

        items, kwargs = [], {"KeyConditionExpression": Key("run_id").eq(run_id)}
        while True:
            page = probes_table.query(**kwargs)
            items += page["Items"]
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return [{"probe_id": i["probe_id"], "set": json.loads(i["set"]), "map50": float(i["map50"])}
            for i in items]


def _emit_metrics(arch: str, ledger_size: int, step: dict) -> None:
    """CloudWatch Embedded Metric Format: one structured log line per round.

    CloudWatch extracts the metrics from the Lambda's log, so the function
    needs no PutMetricData permission and no metrics client.
    """
    located = sum(1 for f in step.get("findings", []) if f.get("status") == "located")
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "Blindspot", "Dimensions": [["Arch"]],
            "Metrics": [{"Name": "ProbesCompleted", "Unit": "Count"},
                        {"Name": "SpentUSD", "Unit": "None"},
                        {"Name": "AxesLocated", "Unit": "Count"},
                        {"Name": "WaveSize", "Unit": "Count"},
                        {"Name": "RunHalted", "Unit": "Count"}]}]},
        "Arch": arch, "ProbesCompleted": ledger_size,
        "SpentUSD": step.get("spent_usd", 0.0), "AxesLocated": located,
        "WaveSize": len(step.get("probes", [])),
        "RunHalted": 1 if step.get("action") == "halted" else 0,
    }))


def _decimalise(obj):
    return json.loads(json.dumps(obj), parse_float=Decimal)


def handler(event, context):
    runs, probes, decisions, s3, bucket = _clients()
    run_id, arch, mode = event["run_id"], event.get("arch", "arm64"), event["mode"]
    run_item = runs.get_item(Key={"run_id": run_id})["Item"]
    run = json.loads(run_item["definition"])
    base = {"run_id": run_id, "arch": arch, "size": 0, "spec": ""}

    if mode == "halt":
        runs.put_item(Item={**run_item, "status": "AWAITING_APPROVAL", "updated": int(time.time())})
        return {**base, "action": "halted"}

    ledger = _ledger(probes, run_id)
    step = plan_round(run, ledger)
    if mode == "plan":
        _emit_metrics(arch, len(ledger), step)

    if mode == "finalize":
        key = f"runs/{run_id}/envelope.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(step, indent=1),
                      ContentType="application/json")
        runs.put_item(Item={**run_item, "status": "DONE", "envelope": f"s3://{bucket}/{key}",
                            "updated": int(time.time())})
        return {**base, "action": "done"}

    round_no = int(run_item.get("round", 0)) + 1
    decisions.put_item(Item=_decimalise({
        "run_id": run_id, "decision_id": f"{round_no:04d}-plan",
        "tool": "scheduler.plan_round",
        "input": {"ledger_size": len(ledger)},
        "output": {k: v for k, v in step.items() if k != "probes"} | {
            "probe_ids": [p["probe_id"] for p in step.get("probes", [])]},
        "rationale": "deterministic replay of the local search under the budget contract",
        "accepted_by_scheduler": True,
    }))
    runs.put_item(Item={**run_item, "round": round_no, "updated": int(time.time())})

    if step["action"] != "probe":
        return {**base, "action": step["action"]}

    key = f"runs/{run_id}/wave-{round_no:04d}.json"
    wave = {"run_id": run_id, "dataset": run["dataset"], "pipeline": run["pipeline"],
            "seed": run["seed"], "frames": run.get("frames"), "probes": step["probes"]}
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(wave), ContentType="application/json")
    return {**base, "action": "probe", "size": len(step["probes"]), "spec": f"s3://{bucket}/{key}"}
