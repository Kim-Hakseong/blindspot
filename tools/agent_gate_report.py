"""Summarise the agent's live runs for the W6 gate, from exported ledgers.

Reads bench/out/cloud_runs/<run>-decisions.json (tools/export_decisions.py) for
every run that has an agent.summary record and writes
bench/out/cloud_runs/agent_gate.json: each proposal with the gate's verdict
and reason, each model call with its charge, and how the run ended.

    uv run python tools/agent_gate_report.py
"""

from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "bench" / "out" / "cloud_runs"
PROPOSALS = {"reallocate_budget", "propose_axis_priority"}


def summarise(decisions: list[dict]) -> dict:
    summary = next(d for d in decisions if d["tool"] == "agent.summary")["output"]
    proposals = [{"tool": d["tool"], "accepted": d["accepted_by_scheduler"],
                  "probes": sum(d["input"].get("probes_per_axis", {}).values()) or None,
                  "reason": d["output"].get("reason"), "rationale": d["rationale"]}
                 for d in decisions if d["tool"] in PROPOSALS]
    calls = [{"call": int(d["input"]["call"]), "usd": d["output"]["usd"],
              "charged": d["accepted_by_scheduler"],
              "contract_spent_after_usd": d["output"]["contract_spent_after_usd"]}
             for d in decisions if d["tool"] == "agent.model_call"]
    approvals = [{"approver": d["input"].get("approver"), "channel": d["input"].get("channel"),
                  "contract_version": d["output"].get("contract_version"),
                  "applied_allocation": d["output"].get("applied_allocation") is not None}
                 for d in decisions if d["tool"] == "human.approve"]
    cancels = [d["rationale"] for d in decisions if d["tool"] == "operator.cancel"]
    return {
        "model": summary["model"], "model_calls": int(summary["model_calls"]),
        "model_usd": summary["model_usd"], "run_status_after_agent": summary["run_status"],
        "proposals": proposals,
        "proposals_accepted": sum(p["accepted"] for p in proposals),
        "proposals_refused": sum(not p["accepted"] for p in proposals),
        "per_call_charges": calls, "per_call_charges_sum_usd": sum(c["usd"] for c in calls) if calls else None,
        "approvals": approvals, "cancelled": cancels,
    }


def main() -> int:
    runs = {}
    for f in sorted(OUT.glob("*-decisions.json")):
        decisions = json.loads(f.read_text())
        if any(d["tool"] == "agent.summary" for d in decisions):
            runs[f.name[: -len("-decisions.json")]] = summarise(decisions)
    report = {"runs": runs,
              "totals": {"runs": len(runs),
                         "proposals_accepted": sum(r["proposals_accepted"] for r in runs.values()),
                         "proposals_refused": sum(r["proposals_refused"] for r in runs.values()),
                         "model_usd": sum(r["model_usd"] for r in runs.values())},
              "command": "uv run python tools/agent_gate_report.py"}
    (OUT / "agent_gate.json").write_text(json.dumps(report, indent=1) + "\n")
    for run, r in runs.items():
        print(f"{run}: {r['model_calls']} calls ${r['model_usd']:.4f}, proposals "
              f"{r['proposals_accepted']} accepted / {r['proposals_refused']} refused, "
              f"after agent: {r['run_status_after_agent']}, approvals {len(r['approvals'])}, "
              f"cancelled {bool(r['cancelled'])}")
    print(report["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
