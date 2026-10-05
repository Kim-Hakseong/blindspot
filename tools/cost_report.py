"""Measured cost of one cloud run.

Collects the run's billable usage from AWS and prices it with
`blindspot.cost.ledger` (pure arithmetic, testable by hand):

- Fargate: every Batch job the run's state machine submitted, array children
  included; billable time is ECS's own pullStartedAt -> stoppedAt for the task,
  which is what Fargate bills (image pull included, one-minute minimum).
- Step Functions: state transitions in the execution history.
- Lambda: planner invocations, timed from the execution history (an upper
  bound on billed duration).

ECS keeps stopped tasks for about an hour, so run this soon after the run.

    AWS_PROFILE=blindspot uv run --group cloud python tools/cost_report.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import boto3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.cost.ledger import FargateTask, run_cost  # noqa: E402

LAMBDA_MEMORY_GB = 0.25


def stack_outputs(cf, stack: str) -> dict:
    return {o["OutputKey"]: o["OutputValue"]
            for o in cf.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}


def history(sfn, execution_arn: str) -> list[dict]:
    events, token = [], None
    while True:
        kw = {"executionArn": execution_arn, "maxResults": 1000}
        if token:
            kw["nextToken"] = token
        page = sfn.get_execution_history(**kw)
        events += page["events"]
        token = page.get("nextToken")
        if not token:
            return events


def batch_job_ids(events: list[dict]) -> list[str]:
    ids = []
    for e in events:
        details = e.get("taskSucceededEventDetails") or e.get("taskFailedEventDetails") or {}
        if details.get("resourceType") == "batch":
            out = json.loads(details.get("output") or "{}")
            if out.get("JobId"):
                ids.append(out["JobId"])
    return ids


def expand_arrays(batch, job_ids: list[str]) -> list[dict]:
    """Describe the submitted jobs, replacing each array parent by its children."""
    jobs = []
    for i in range(0, len(job_ids), 100):
        jobs += batch.describe_jobs(jobs=job_ids[i:i + 100])["jobs"]
    out = []
    for job in jobs:
        if not job.get("arrayProperties", {}).get("size"):
            out.append(job)
            continue
        child_ids = []
        for status in ("SUCCEEDED", "FAILED"):
            token = None
            while True:
                kw = {"arrayJobId": job["jobId"], "jobStatus": status}
                if token:
                    kw["nextToken"] = token
                page = batch.list_jobs(**kw)
                child_ids += [j["jobId"] for j in page["jobSummaryList"]]
                token = page.get("nextToken")
                if not token:
                    break
        for k in range(0, len(child_ids), 100):
            out += batch.describe_jobs(jobs=child_ids[k:k + 100])["jobs"]
    return out


def fargate_tasks(batch, ecs, jobs: list[dict]) -> tuple[list[FargateTask], list[str]]:
    clusters: dict[str, str] = {}
    by_cluster: dict[str, list[tuple[str, str]]] = {}
    missing = []
    for job in jobs:
        queue = job["jobQueue"].rsplit("/", 1)[-1]
        arch = "x86" if queue.endswith("x86") else "arm64"
        if queue not in clusters:
            q = batch.describe_job_queues(jobQueues=[job["jobQueue"]])["jobQueues"][0]
            ce = q["computeEnvironmentOrder"][0]["computeEnvironment"]
            clusters[queue] = batch.describe_compute_environments(
                computeEnvironments=[ce])["computeEnvironments"][0]["ecsClusterArn"]
        for attempt in job.get("attempts", []):
            arn = attempt.get("container", {}).get("taskArn")
            if arn:
                by_cluster.setdefault(clusters[queue], []).append((arn, arch))
            else:
                missing.append(job["jobId"])
    tasks = []
    for cluster, items in by_cluster.items():
        arch_of = dict(items)
        arns = [a for a, _ in items]
        for i in range(0, len(arns), 100):
            for t in ecs.describe_tasks(cluster=cluster, tasks=arns[i:i + 100])["tasks"]:
                start = t.get("pullStartedAt") or t.get("startedAt")
                stop = t.get("stoppedAt") or t.get("executionStoppedAt")
                if not (start and stop):
                    missing.append(t["taskArn"])
                    continue
                tasks.append(FargateTask(arch=arch_of[t["taskArn"]], vcpu=int(t["cpu"]) / 1024,
                                         memory_gb=int(t["memory"]) / 1024,
                                         seconds=(stop - start).total_seconds()))
            # Tasks ECS no longer reports (stopped > ~1 h ago) are listed as missing.
    return tasks, missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--stack", default="Blindspot")
    p.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "bench" / "out" / "cloud_runs")
    a = p.parse_args()

    session = boto3.Session()
    out = stack_outputs(session.client("cloudformation"), a.stack)
    sfn, batch, ecs = session.client("stepfunctions"), session.client("batch"), session.client("ecs")

    executions = [e for e in sfn.list_executions(stateMachineArn=out["StateMachineArn"],
                                                 maxResults=100)["executions"]
                  if e["name"] == a.run_id or e["name"].startswith(a.run_id + "-v")]
    if not executions:
        sys.exit(f"no execution named {a.run_id}")

    all_events, transitions, lambda_s, lambda_n, job_ids = [], 0, 0.0, 0, []
    for ex in executions:
        events = history(sfn, ex["executionArn"])
        all_events += events
        transitions += sum(1 for e in events if e["type"].endswith("StateEntered"))
        scheduled = {}
        for e in events:
            if e["type"] == "TaskScheduled" and e["taskScheduledEventDetails"]["resourceType"] == "lambda":
                scheduled[e["id"]] = e["timestamp"]
            if e["type"] == "TaskSucceeded" and e["taskSucceededEventDetails"]["resourceType"] == "lambda":
                start = max((t for i, t in scheduled.items() if i < e["id"]), default=None)
                if start:
                    lambda_s += (e["timestamp"] - start).total_seconds()
                    lambda_n += 1
        job_ids += batch_job_ids(events)

    jobs = expand_arrays(batch, job_ids)
    tasks, missing = fargate_tasks(batch, ecs, jobs)
    cost = run_cost(tasks, lambda_gb_seconds=lambda_s * LAMBDA_MEMORY_GB,
                    lambda_requests=lambda_n, sfn_transitions=transitions)
    report = {
        "run_id": a.run_id,
        "executions": [{"name": e["name"], "status": e["status"],
                        "start": e["startDate"].isoformat(),
                        "stop": e.get("stopDate").isoformat() if e.get("stopDate") else None}
                       for e in executions],
        "batch_jobs": len(jobs),
        "tasks_unaccounted": missing,
        **cost,
        "command": f"AWS_PROFILE=blindspot uv run --group cloud python tools/cost_report.py --run-id {a.run_id}",
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.run_id}-cost.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"run {a.run_id}: total {report['total_usd']:.4f} USD "
          f"(Fargate {report['fargate_usd']:.4f} over {report['fargate_tasks']} tasks, "
          f"{report['fargate_billed_seconds']:.0f} billed s; Lambda {report['lambda_usd']:.5f}; "
          f"Step Functions {report['sfn_usd']:.5f})")
    if missing:
        print(f"WARNING: {len(missing)} task(s) not priced (ECS no longer reports them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
