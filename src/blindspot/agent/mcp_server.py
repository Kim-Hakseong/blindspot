"""MCP server exposing Blindspot to an agent.

Tools, and what each is allowed to do (rule A1):

  get_envelope           read-only: findings, coverage, budget state
  explain_failure        read-only: the *facts* behind a boundary; the model
                         writes the prose, the tool never does
  probe_condition        run one measurement under the budget contract; pass
                         or fail is applied by boundary.criterion, not by the
                         caller
  propose_axis_priority  a proposal -> cost.gate decides
  reallocate_budget      a proposal -> cost.gate decides; beyond the contract
                         the run halts in AWAITING_APPROVAL

Every call is written to the decision ledger (rule A3), rejected ones
included. Proposals count toward the per-run call cap (rule A5).

    uv run --group agent python -m blindspot.agent.mcp_server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from dataclasses import dataclass, field

from mcp.server.mcpserver import MCPServer

from ..cost.contract import BudgetContract, RunState
from ..cost.gate import Proposal, heuristic_order, review
from .ledger import DecisionLedger

ROOT = pathlib.Path(__file__).resolve().parents[3]


@dataclass
class AgentSession:
    dataset: pathlib.Path
    pipeline: str
    frames: int | None
    seed: int
    axes: list[str]
    coverage: dict[str, float]
    baseline_map50: float
    threshold_map50: float
    budget_usd: float
    cost_per_probe_usd: float
    ledger_path: pathlib.Path
    findings: list[dict] = field(default_factory=list)
    order: list[str] | None = None
    allocation: dict[str, int] | None = None
    proposals: int = 0
    #: A split the gate refused for exceeding the contract, kept for the human.
    pending_allocation: dict[str, int] | None = None
    #: False for a cloud run: measurements happen on AWS Batch, so the agent
    #: is not offered probe_condition and only reads and proposes.
    measure_locally: bool = True
    #: Any object with record(entry); a JSONL file by default, bs-decisions in the cloud.
    ledger_sink: object | None = None

    def __post_init__(self):
        self.contract = BudgetContract(self.budget_usd, self.cost_per_probe_usd)
        self.ledger = self.ledger_sink or DecisionLedger(self.ledger_path)
        self._runtime = None
        if self.order is None:
            self.order = heuristic_order(self.axes, self.coverage)

    def runtime(self):
        if self._runtime is None:
            from ..runner.dataset import ValidationSet
            from ..runner.registry import load_pipeline
            from ..runner.yolox import COCO_CLASSES

            vs = ValidationSet(self.dataset, COCO_CLASSES)
            self._runtime = (vs, vs.load(limit=self.frames), load_pipeline(self.pipeline))
        return self._runtime


def build_server(s: AgentSession) -> MCPServer:
    server = MCPServer(
        name="blindspot",
        instructions=(
            "Blindspot finds where a vision pipeline starts failing. You may propose the "
            "order of axes and a split of the remaining probe budget; deterministic code "
            "accepts or rejects each proposal. You never decide pass/fail or boundaries."
        ),
    )

    def log(tool: str, args: dict, output: dict, rationale: str, accepted: bool) -> dict:
        s.ledger.record({"tool": tool, "input": args, "output": output,
                         "rationale": rationale, "accepted_by_scheduler": accepted})
        return output

    @server.tool(description="Current findings, coverage and budget. Read-only.")
    def get_envelope() -> dict:
        out = {"findings": s.findings, "coverage_percent": s.coverage, "axis_order": s.order,
               "allocation": s.allocation, "budget": s.contract.describe(),
               "threshold_map50": s.threshold_map50, "baseline_map50": s.baseline_map50}
        return log("get_envelope", {}, out, "read-only", True)

    @server.tool(description="Facts behind the boundary on one axis, for you to explain. "
                             "Returns data only; writing the explanation is your job.")
    def explain_failure(axis: str) -> dict:
        f = next((x for x in s.findings if x["axis"] == axis), None)
        out = {"axis": axis,
               "boundary": {"lower": f["lower"], "upper": f["upper"]} if f else None,
               "coverage_percent": s.coverage.get(axis),
               "criterion": f"mAP@50 below {s.threshold_map50:.4f} (60% of baseline {s.baseline_map50:.4f})"}
        return log("explain_failure", {"axis": axis}, out, "read-only", True)

    @server.tool(description="Measure one single-axis condition. Charged against the budget "
                             "contract; refused with AWAITING_APPROVAL when it cannot pay.")
    def probe_condition(axis: str, value: float) -> dict:
        args = {"axis": axis, "value": value}
        if axis not in s.axes:
            return log("probe_condition", args, {"error": f"{axis} is not an axis of this run"}, "", False)
        decision = s.contract.request(1)
        if not decision.approved:
            return log("probe_condition", args,
                       {"state": RunState.AWAITING_APPROVAL.value, "reason": decision.reason}, "", False)
        s.contract.charge(1)
        from ..boundary.probe import run_condition_probe
        from ..degrade.compose import Condition

        vs, frames, pipeline = s.runtime()
        scored = run_condition_probe(frames, pipeline, vs, Condition.from_axes({axis: value}), s.seed)
        out = {"axis": axis, "value": value, "map50": scored["map50"],
               "failed": scored["map50"] < s.threshold_map50,
               "judged_by": "boundary.criterion (deterministic)",
               "budget": s.contract.describe(),
               "reproduce": f"uv run blindspot probe --set {axis}={float(value)!r} --seed {s.seed}"}
        return log("probe_condition", args, out, "measurement", True)

    def propose(tool: str, gate_tool: str, args: dict, rationale: str) -> dict:
        verdict = review(Proposal(gate_tool, args, rationale), s.axes, s.contract, s.proposals)
        s.proposals += 1
        if verdict.accepted:
            if gate_tool == "prioritize_axes":
                s.order = verdict.applied["order"]
            else:
                s.allocation = verdict.applied["probes_per_axis"]
        elif verdict.state is RunState.AWAITING_APPROVAL:
            s.pending_allocation = dict(args["probes_per_axis"])
        record = verdict.to_record() | {"tool": tool}
        s.ledger.record(record)
        return record

    @server.tool(description="Propose the order in which axes are probed. The scheduler decides.")
    def propose_axis_priority(order: list[str], rationale: str) -> dict:
        return propose("propose_axis_priority", "prioritize_axes", {"order": order}, rationale)

    @server.tool(description="Propose how the remaining probes are split across axes. "
                             "Anything beyond the budget contract halts for human approval.")
    def reallocate_budget(probes_per_axis: dict[str, int], rationale: str) -> dict:
        return propose("reallocate_budget", "reallocate_budget",
                       {"probes_per_axis": probes_per_axis}, rationale)

    return server


def session_from_report(report_path: pathlib.Path, ledger: pathlib.Path, frames: int | None,
                        budget_probes: int) -> AgentSession:
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    if report is None:
        eff = json.loads((ROOT / "bench" / "out" / "efficiency.json").read_text())
        axes = [a["axis"] for a in eff["axes"]]
        return AgentSession(ROOT / "val" / "road100", "yolox_s", frames, eff["seed"], axes, {},
                            eff["baseline"]["mAP50"], eff["criterion"]["threshold_map50"],
                            budget_probes * 0.01, 0.01, ledger, [])
    run = report["run"]
    return AgentSession(
        dataset=ROOT / "val" / run["dataset"], pipeline=run["pipeline"], frames=frames, seed=run["seed"],
        axes=[f["axis"] for f in report["findings"]],
        coverage={r["axis"]: r["coverage_percent"] for r in report["uncovered_regions"]},
        baseline_map50=run["baseline_map50"], threshold_map50=run["threshold_map50"],
        budget_usd=budget_probes * 0.01, cost_per_probe_usd=0.01, ledger_path=ledger,
        findings=[{k: f[k] for k in ("axis", "unit", "status", "lower", "upper")} for f in report["findings"]],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=pathlib.Path, default=ROOT / "viewer" / "public" / "data" / "report.json")
    p.add_argument("--ledger", type=pathlib.Path, default=ROOT / "runs" / "decisions.jsonl")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--budget-probes", type=int, default=40)
    a = p.parse_args()
    asyncio.run(build_server(session_from_report(a.report, a.ledger, a.frames, a.budget_probes))
                .run_stdio_async())


if __name__ == "__main__":
    main()
