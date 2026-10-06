"""Print one run's decision ledger (bs-decisions) as a readable trace.

    AWS_PROFILE=blindspot uv run --group cloud python tools/show_decisions.py --run-id <id>

What the agent read (boundaries measured by OpenCV 5 degradations and the
detector), each model call and what it cost the contract, each proposal and the
gate's verdict, and the planner's rounds that followed.
"""

from __future__ import annotations

import argparse


def _fmt(x) -> str:
    return f"{float(x):g}"


def line(rec: dict) -> str:
    tool, out, inp = rec["tool"], rec.get("output", {}), rec.get("input", {})
    ok = "accepted" if rec.get("accepted_by_scheduler") else "REFUSED "
    head = f"{rec['decision_id']:<16} {tool:<24} {ok}"
    if tool == "get_envelope":
        bounds = "; ".join(f"{f['axis']} {_fmt(f['lower'])}–{_fmt(f['upper'])} {f['unit']}"
                           for f in out.get("findings", []) if f.get("lower") is not None)
        return f"{head} read measured boundaries: {bounds}"
    if tool == "explain_failure":
        return f"{head} facts for {inp.get('axis')}"
    if tool == "agent.model_call":
        return (f"{head} call {int(inp.get('call', 0))}: charged ${_fmt(out.get('usd', 0))}, "
                f"contract left ${_fmt(out.get('contract_remaining_usd', 0))}")
    if tool in ("reallocate_budget", "propose_axis_priority"):
        size = sum(inp.get("probes_per_axis", {}).values())
        what = f"{int(size)} probes split" if size else f"order {inp.get('order')}"
        return f"{head} {what} -> {out.get('reason', '')}"
    if tool == "agent.summary":
        return (f"{head} {int(out.get('model_calls', 0))} model calls, ${_fmt(out.get('model_usd', 0))}; "
                f"run {out.get('run_status')}")
    if tool == "scheduler.plan_round":
        return f"{head} round: {out.get('action')} {len(out.get('probe_ids', []))} probes"
    if tool in ("human.approve", "operator.cancel"):
        return f"{head} {inp.get('approver') or inp.get('operator')}: {rec.get('rationale', '')[:80]}"
    return head


def main() -> int:
    import boto3
    from boto3.dynamodb.conditions import Key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    items = boto3.resource("dynamodb").Table("bs-decisions").query(
        KeyConditionExpression=Key("run_id").eq(args.run_id))["Items"]
    for rec in sorted(items, key=lambda r: r["decision_id"])[: args.limit]:
        print(line(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
