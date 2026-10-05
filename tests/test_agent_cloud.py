"""The agent before a cloud run (W6, reusing the W4 halt/approval path).

`blindspot cloud-run --agent` lets the agent propose an axis order and a
per-axis probe split before the Step Functions loop starts. What reaches the
run is decided here, deterministically, from the gate's verdicts:

- accepted order and split are written into the run definition;
- the agent's measured model cost is charged to the run's own contract;
- a split the contract cannot pay for, after that charge, is not applied: the
  run is created AWAITING_APPROVAL and `blindspot approve` resumes it;
- every agent decision lands in bs-decisions under the run's id.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from blindspot.cloud.agent_plan import DynamoDecisionLedger, prepare_agent_run

AXES = ["motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m", "jpeg.quality"]


class Table:
    def __init__(self):
        self.items = []

    def put_item(self, Item):
        self.items.append(Item)


def definition(budget=0.40, cost=0.002):
    return {"run_id": "r1", "budget_usd": budget, "cost_per_probe_usd": cost,
            "axes": [{"axis": a, "unit": "u"} for a in AXES]}


def outcome(order=None, allocation=None, pending=None, usd=0.01):
    return {"axis_order": order or AXES, "allocation": allocation,
            "pending_allocation": pending, "model_usd": usd, "status": "completed",
            "model": "m", "model_calls": 3}


def test_decisions_go_to_the_table_under_the_run_id_in_order():
    table = Table()
    ledger = DynamoDecisionLedger(table, "r1")
    ledger.record({"tool": "get_envelope", "input": {}, "output": {"x": 0.5},
                   "rationale": "read-only", "accepted_by_scheduler": True})
    ledger.record({"tool": "reallocate_budget", "input": {}, "output": {},
                   "rationale": "r", "accepted_by_scheduler": False})
    assert [i["decision_id"] for i in table.items] == ["0000-agent-001", "0000-agent-002"]
    assert all(i["run_id"] == "r1" for i in table.items)
    assert isinstance(table.items[0]["output"]["x"], Decimal)  # DynamoDB has no floats
    assert table.items[1]["accepted_by_scheduler"] is False


def test_accepted_proposals_are_written_into_the_definition():
    order = list(reversed(AXES))
    d, status = prepare_agent_run(definition(), outcome(order=order, allocation={AXES[0]: 10}))
    assert status == "RUNNING"
    assert [a["axis"] for a in d["axes"]] == order
    assert d["probes_per_axis"] == {AXES[0]: 10}
    assert d["prepaid_usd"] == pytest.approx(0.01)


def test_a_split_beyond_the_contract_creates_the_run_awaiting_approval():
    d, status = prepare_agent_run(definition(), outcome(pending={AXES[0]: 500}))
    assert status == "AWAITING_APPROVAL"
    assert d["pending_allocation"] == {AXES[0]: 500} and "probes_per_axis" not in d


def test_a_split_the_model_cost_made_unaffordable_is_held_back_too():
    # 0.02 at 0.002/probe pays 10 probes; 0.01 spent on the model leaves 5.
    d, status = prepare_agent_run(definition(budget=0.02),
                                  outcome(allocation={AXES[0]: 8}, usd=0.01))
    assert status == "AWAITING_APPROVAL"
    assert d["pending_allocation"] == {AXES[0]: 8} and "probes_per_axis" not in d


def test_model_cost_beyond_the_contract_halts_before_any_probe():
    d, status = prepare_agent_run(definition(budget=0.005), outcome(usd=0.01))
    assert status == "AWAITING_APPROVAL" and d["prepaid_usd"] == pytest.approx(0.01)


def test_an_order_that_is_not_a_permutation_is_ignored():
    d, _ = prepare_agent_run(definition(), outcome(order=AXES[:2]))
    assert [a["axis"] for a in d["axes"]] == AXES
