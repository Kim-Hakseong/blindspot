"""The gate between an agent's proposals and the scheduler.

An agent can propose two things (rule A1): the order in which axes are probed,
and how the remaining probe budget is split across axes. This module decides
whether a proposal is applied. It is deterministic and sits in a judgment
package, so it cannot import a model client (rule A2, enforced by
tests/test_no_llm_in_judgment.py).

- A proposal outside those two capabilities is rejected, whatever its reasons.
- An allocation larger than the contract can pay for is not applied; the run
  halts in AWAITING_APPROVAL (rule A4).
- After LLM_CALL_CAP proposals the gate rejects further ones and the run
  continues on a fixed heuristic (rule A5).

Every verdict serialises to the decision-ledger record shape (rule A3),
accepted or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract import BudgetContract, RunState

LLM_CALL_CAP = 10
CAPABILITIES = {"prioritize_axes", "reallocate_budget"}


@dataclass(frozen=True)
class Proposal:
    tool: str
    args: dict
    rationale: str


@dataclass(frozen=True)
class Verdict:
    proposal: Proposal
    accepted: bool
    state: RunState
    reason: str
    applied: dict | None = field(default=None)

    def to_record(self) -> dict:
        return {
            "tool": self.proposal.tool,
            "input": self.proposal.args,
            "output": {"state": self.state.value, "reason": self.reason, "applied": self.applied},
            "rationale": self.proposal.rationale,
            "accepted_by_scheduler": self.accepted,
        }


def _reject(p: Proposal, reason: str, state: RunState = RunState.RUNNING) -> Verdict:
    return Verdict(p, False, state, reason)


def heuristic_order(axes: list[str], coverage_percent: dict[str, float]) -> list[str]:
    """Fallback order: least-covered axis first, ties by name. No model involved."""
    return sorted(axes, key=lambda a: (coverage_percent.get(a, 0.0), a))


def review(p: Proposal, run_axes: list[str], contract: BudgetContract, llm_calls_used: int) -> Verdict:
    if llm_calls_used >= LLM_CALL_CAP:
        return _reject(p, f"LLM call cap ({LLM_CALL_CAP}) reached; continuing on the fixed heuristic")
    if p.tool not in CAPABILITIES:
        return _reject(p, f"{p.tool!r} is not an agent capability; allowed: {sorted(CAPABILITIES)}")

    if p.tool == "prioritize_axes":
        order = list(p.args.get("order", []))
        if sorted(order) != sorted(run_axes) or len(set(order)) != len(order):
            return _reject(p, "order must be a permutation of this run's axes")
        return Verdict(p, True, RunState.RUNNING, "order applied", {"order": order})

    alloc = p.args.get("probes_per_axis", {})
    if not isinstance(alloc, dict) or not alloc:
        return _reject(p, "probes_per_axis must be a non-empty mapping")
    for axis, n in alloc.items():
        if axis not in run_axes:
            return _reject(p, f"{axis!r} is not an axis of this run")
        if not isinstance(n, int) or n < 0:
            return _reject(p, f"allocation for {axis!r} must be a non-negative integer")
    requested = sum(alloc.values())
    decision = contract.request(requested)
    if not decision.approved:
        return _reject(p, decision.reason, RunState.AWAITING_APPROVAL)
    return Verdict(p, True, RunState.RUNNING, "allocation applied", {"probes_per_axis": dict(alloc)})
