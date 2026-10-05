"""Human approval of a halted run (W6-4, human control).

Code running under a contract cannot raise it. Continuing a halted run is a
separate human act that records who approved how much and why, as a new
contract version, before the loop restarts.
"""

import json

import pytest

from blindspot.cloud.approve import approve


class Table:
    def __init__(self, items=None):
        self.items = dict(items or {})

    def get_item(self, Key):
        k = (Key["run_id"], "")
        return {"Item": self.items[k]} if k in self.items else {}

    def put_item(self, Item):
        self.items[(Item["run_id"], Item.get("decision_id", ""))] = Item


def halted_run():
    return Table({("r1", ""): {"run_id": "r1", "status": "AWAITING_APPROVAL", "arch": "arm64",
                               "definition": json.dumps({"budget_usd": 0.08, "run_id": "r1"})}})


def test_approval_raises_the_limit_as_a_recorded_new_contract():
    runs, decisions, started = halted_run(), Table(), []
    out = approve(runs, decisions, lambda **kw: started.append(kw), "r1", 0.10, "haku", "need jpeg axis")
    item = runs.items[("r1", "")]
    assert json.loads(item["definition"])["budget_usd"] == pytest.approx(0.18)
    assert item["status"] == "RUNNING" and int(item["contract_version"]) == 2
    rec = next(v for (r, d), v in decisions.items.items() if d)
    assert rec["tool"] == "human.approve" and rec["accepted_by_scheduler"] is True
    # Stored as Decimal, which DynamoDB requires.
    assert float(rec["input"]["additional_usd"]) == pytest.approx(0.10)
    assert rec["input"]["approver"] == "haku"
    assert rec["rationale"] == "need jpeg axis"
    assert started and out["budget_usd"] == pytest.approx(0.18)


def test_only_a_halted_run_can_be_approved():
    runs = halted_run()
    runs.items[("r1", "")]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="AWAITING_APPROVAL"):
        approve(runs, Table(), lambda **kw: None, "r1", 0.1, "haku", "x")


@pytest.mark.parametrize("amount", [0, -0.1])
def test_approval_must_add_a_positive_amount(amount):
    with pytest.raises(ValueError):
        approve(halted_run(), Table(), lambda **kw: None, "r1", amount, "haku", "x")


def test_approval_needs_a_named_approver_and_a_reason():
    with pytest.raises(ValueError):
        approve(halted_run(), Table(), lambda **kw: None, "r1", 0.1, "", "x")
    with pytest.raises(ValueError):
        approve(halted_run(), Table(), lambda **kw: None, "r1", 0.1, "haku", "  ")


def test_approval_applies_a_pending_agent_allocation_once_it_fits():
    # The agent proposed 12 probes; the 0.08 contract at 0.01/probe could not pay,
    # so the run waited. Approval that makes it affordable applies it.
    d = {"budget_usd": 0.08, "cost_per_probe_usd": 0.01, "run_id": "r1",
         "pending_allocation": {"a": 7, "b": 5}}
    runs = Table({("r1", ""): {"run_id": "r1", "status": "AWAITING_APPROVAL", "arch": "arm64",
                               "definition": json.dumps(d)}})
    decisions = Table()
    approve(runs, decisions, lambda **kw: None, "r1", 0.05, "haku", "agent split is sensible")
    after = json.loads(runs.items[("r1", "")]["definition"])
    assert after["probes_per_axis"] == {"a": 7, "b": 5} and "pending_allocation" not in after
    rec = next(v for (r, k), v in decisions.items.items() if k)
    assert rec["output"]["applied_allocation"] == {"a": 7, "b": 5}


def test_approval_too_small_for_the_pending_allocation_keeps_it_pending():
    d = {"budget_usd": 0.08, "cost_per_probe_usd": 0.01, "run_id": "r1",
         "pending_allocation": {"a": 20}}
    runs = Table({("r1", ""): {"run_id": "r1", "status": "AWAITING_APPROVAL", "arch": "arm64",
                               "definition": json.dumps(d)}})
    approve(runs, Table(), lambda **kw: None, "r1", 0.02, "haku", "partial")
    after = json.loads(runs.items[("r1", "")]["definition"])
    assert "probes_per_axis" not in after and after["pending_allocation"] == {"a": 20}
