"""Cancelling a run that should never resume (operator action, recorded).

Only a run that is not executing can be cancelled: AWAITING_APPROVAL (or
PLANNING) runs. The status becomes CANCELLED, the reason is written to the
decision ledger, and approve() refuses it afterwards. Who cancelled follows
the approval rule: without a terminal it is recorded as automation.
"""

import json

import pytest

from blindspot.cloud.approve import approve
from blindspot.cloud.cancel import cancel


class Table:
    def __init__(self, items=None):
        self.items = dict(items or {})

    def get_item(self, Key):
        k = (Key["run_id"], "")
        return {"Item": self.items[k]} if k in self.items else {}

    def put_item(self, Item):
        self.items[(Item["run_id"], Item.get("decision_id", ""))] = Item


def run(status="AWAITING_APPROVAL"):
    return Table({("r1", ""): {"run_id": "r1", "status": status, "arch": "arm64",
                               "definition": json.dumps({"budget_usd": 0.4, "cost_per_probe_usd": 0.002})}})


def test_a_halted_run_is_cancelled_with_its_reason_in_the_ledger():
    runs, decisions = run(), Table()
    cancel(runs, decisions, "r1", "stopped by a fixed bug", operator="haku", interactive=False)
    assert runs.items[("r1", "")]["status"] == "CANCELLED"
    rec = next(v for (r, d), v in decisions.items.items() if d)
    assert rec["tool"] == "operator.cancel" and rec["rationale"] == "stopped by a fixed bug"
    assert rec["input"]["operator"] == "automation" and rec["input"]["operator_claimed"] == "haku"


@pytest.mark.parametrize("status", ["RUNNING", "DONE", "CANCELLED"])
def test_only_a_run_that_is_not_executing_can_be_cancelled(status):
    with pytest.raises(ValueError):
        cancel(run(status), Table(), "r1", "x", operator="a", interactive=False)


def test_a_cancelled_run_cannot_be_approved():
    runs = run()
    cancel(runs, Table(), "r1", "x", operator="a", interactive=False)
    with pytest.raises(ValueError, match="CANCELLED"):
        approve(runs, Table(), lambda **k: None, "r1", 0.1, "a", "b")


def test_a_reason_is_required():
    with pytest.raises(ValueError):
        cancel(run(), Table(), "r1", "  ", operator="a", interactive=False)
