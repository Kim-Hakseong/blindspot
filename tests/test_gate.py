"""The gate between an agent's proposals and the scheduler (rules A1-A5).

The agent may propose an axis order and a reallocation of the remaining
budget. The gate is deterministic code in a judgment package, so it cannot
consult a model; it accepts, rejects, or halts for approval, and every one of
those outcomes is something the ledger can record.
"""

from __future__ import annotations

import pytest

from blindspot.cost.contract import BudgetContract, RunState
from blindspot.cost.gate import LLM_CALL_CAP, Proposal, heuristic_order, review

AXES = ["motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m"]


def contract(spent_probes=0, limit=0.40):
    c = BudgetContract(limit, 0.01)
    if spent_probes:
        c.charge(spent_probes)
    return c


def test_a_valid_axis_order_is_accepted_and_applied():
    p = Proposal("prioritize_axes", {"order": list(reversed(AXES))}, "fog is least covered")
    v = review(p, AXES, contract(), llm_calls_used=1)
    assert v.accepted and v.applied == {"order": list(reversed(AXES))}


@pytest.mark.parametrize("order", [AXES[:2], AXES + ["jpeg.quality"], [AXES[0]] * 3])
def test_an_order_that_is_not_a_permutation_is_rejected(order):
    v = review(Proposal("prioritize_axes", {"order": order}, "x"), AXES, contract(), 1)
    assert not v.accepted and v.state is RunState.RUNNING


def test_reallocation_within_the_contract_is_accepted():
    p = Proposal("reallocate_budget", {"probes_per_axis": {AXES[0]: 10, AXES[1]: 20}}, "r")
    v = review(p, AXES, contract(spent_probes=5), 1)
    assert v.accepted and v.applied["probes_per_axis"][AXES[1]] == 20


def test_reallocation_beyond_the_contract_halts_for_approval():
    """A4: the agent cannot spend past the contract; the run stops instead."""
    p = Proposal("reallocate_budget", {"probes_per_axis": {AXES[0]: 30, AXES[1]: 30}}, "more")
    v = review(p, AXES, contract(spent_probes=5), 1)
    assert not v.accepted
    assert v.state is RunState.AWAITING_APPROVAL
    assert "contract" in v.reason


@pytest.mark.parametrize("alloc", [{"motion_blur.exposure_ms": -1}, {"not.an_axis": 3}])
def test_malformed_reallocation_is_rejected(alloc):
    v = review(Proposal("reallocate_budget", {"probes_per_axis": alloc}, "x"), AXES, contract(), 1)
    assert not v.accepted and v.state is RunState.RUNNING


@pytest.mark.parametrize("tool", ["set_threshold", "mark_passed", "set_boundary", "raise_budget"])
def test_anything_outside_the_agent_capabilities_is_rejected(tool):
    """A1: the agent proposes order and allocation. It never judges."""
    v = review(Proposal(tool, {}, "trust me"), AXES, contract(), 1)
    assert not v.accepted and "capabilit" in v.reason


def test_the_call_cap_rejects_further_proposals():
    """A5: after the cap the run continues on the fixed heuristic."""
    p = Proposal("prioritize_axes", {"order": AXES}, "x")
    v = review(p, AXES, contract(), llm_calls_used=LLM_CALL_CAP)
    assert not v.accepted and "heuristic" in v.reason


def test_heuristic_probes_the_least_covered_axis_first_and_is_deterministic():
    coverage = {AXES[0]: 17.4, AXES[1]: 88.4, AXES[2]: 30.1}
    assert heuristic_order(AXES, coverage) == [AXES[0], AXES[2], AXES[1]]
    assert heuristic_order(AXES, coverage) == heuristic_order(list(reversed(AXES)), coverage)


def test_every_verdict_is_serialisable_for_the_ledger():
    import json

    v = review(Proposal("prioritize_axes", {"order": AXES}, "r"), AXES, contract(), 1)
    json.dumps(v.to_record())
    assert set(v.to_record()) >= {"tool", "input", "output", "rationale", "accepted_by_scheduler"}
