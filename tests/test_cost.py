"""The budget contract.

A budget that is merely advisory is not a budget. These tests pin the two
properties that make it real: it cannot be raised once a run has started, and
reaching it stops the run rather than overspending or silently truncating.
"""

from __future__ import annotations

import pytest

from blindspot.cost.contract import (
    BudgetContract,
    BudgetExceeded,
    RunState,
)


def contract(limit=0.40, per_probe=0.01):
    return BudgetContract(limit_usd=limit, cost_per_probe_usd=per_probe)


def test_a_fresh_contract_can_afford_a_probe():
    assert contract().can_afford(1)


def test_spending_accumulates():
    c = contract()
    c.charge(3)
    assert c.spent_usd == pytest.approx(0.03)
    assert c.probes_charged == 3


def test_remaining_reflects_spending():
    c = contract()
    c.charge(10)
    assert c.remaining_usd == pytest.approx(0.30)


def test_probes_remaining_is_floor_not_round():
    """0.055 remaining at 0.01 each is 5 probes, not 6."""
    c = BudgetContract(limit_usd=0.055, cost_per_probe_usd=0.01)
    assert c.probes_remaining == 5


def test_charging_past_the_limit_raises():
    c = contract(limit=0.05)
    c.charge(5)
    with pytest.raises(BudgetExceeded):
        c.charge(1)


def test_a_refused_charge_does_not_spend():
    c = contract(limit=0.05)
    c.charge(5)
    with pytest.raises(BudgetExceeded):
        c.charge(1)
    assert c.spent_usd == pytest.approx(0.05)
    assert c.probes_charged == 5


def test_reaching_the_limit_exactly_is_allowed():
    c = contract(limit=0.05)
    c.charge(5)
    assert c.spent_usd == pytest.approx(0.05)
    assert not c.can_afford(1)


def test_the_limit_cannot_be_raised_mid_run():
    c = contract()
    with pytest.raises(AttributeError):
        c.limit_usd = 10.0


def test_requesting_more_than_the_contract_awaits_approval():
    c = contract(limit=0.05)
    c.charge(5)
    decision = c.request(1)
    assert decision.state is RunState.AWAITING_APPROVAL
    assert decision.approved is False
    assert "budget" in decision.reason.lower()


def test_a_request_within_budget_is_granted():
    decision = contract().request(1)
    assert decision.state is RunState.RUNNING
    assert decision.approved is True


def test_a_request_never_charges_by_itself():
    """Asking must not spend; only charging spends."""
    c = contract()
    c.request(5)
    assert c.spent_usd == pytest.approx(0.0)


def test_contract_rejects_a_nonpositive_limit():
    with pytest.raises(ValueError):
        BudgetContract(limit_usd=0.0, cost_per_probe_usd=0.01)


def test_contract_rejects_a_nonpositive_probe_cost():
    with pytest.raises(ValueError):
        BudgetContract(limit_usd=0.4, cost_per_probe_usd=0.0)


def test_ledger_records_every_charge_and_refusal():
    c = contract(limit=0.03)
    c.charge(2)
    with pytest.raises(BudgetExceeded):
        c.charge(5)
    entries = c.ledger
    assert len(entries) == 2
    assert entries[0]["event"] == "charge" and entries[0]["granted"] is True
    assert entries[1]["event"] == "charge" and entries[1]["granted"] is False


def test_partial_results_are_preserved_on_exhaustion():
    """Hitting the cap stops the run; it does not discard what was bought."""
    c = contract(limit=0.03)
    completed = []
    for i in range(10):
        if not c.can_afford(1):
            break
        c.charge(1)
        completed.append(i)
    assert completed == [0, 1, 2]
    assert c.remaining_usd == pytest.approx(0.0)


def test_describe_is_serialisable():
    c = contract()
    c.charge(2)
    described = c.describe()
    assert described["limit_usd"] == pytest.approx(0.40)
    assert described["spent_usd"] == pytest.approx(0.02)
    assert described["probes_charged"] == 2


def test_a_usd_charge_counts_against_the_same_limit():
    from blindspot.cost.contract import BudgetContract, BudgetExceeded
    c = BudgetContract(0.05, 0.01)
    c.charge_usd(0.02, "agent model calls")
    assert c.probes_remaining == 3 and c.ledger[-1]["what"] == "agent model calls"
    with pytest.raises(BudgetExceeded):
        c.charge_usd(0.04, "too much")
    assert c.spent_usd == pytest.approx(0.02)
