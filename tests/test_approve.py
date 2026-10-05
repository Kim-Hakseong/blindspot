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
    out = approve(runs, decisions, lambda **kw: started.append(kw), "r1", 0.10, "haku", "need jpeg axis",
                  confirm=lambda summary: "r1")
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


def test_an_interactive_approval_must_be_confirmed_by_typing_the_run_id():
    runs, decisions, started = halted_run(), Table(), []
    shown = []
    approve(runs, decisions, lambda **kw: started.append(kw), "r1", 0.10, "haku", "ok",
            confirm=lambda summary: shown.append(summary) or "r1")
    rec = next(v for (r, d), v in decisions.items.items() if d)
    assert rec["input"]["approver"] == "haku" and rec["input"]["channel"] == "interactive-confirmed"
    assert "0.08" in shown[0] and "0.18" in shown[0]  # the human sees before and after


def test_a_wrong_confirmation_writes_nothing_and_starts_nothing():
    runs, decisions, started = halted_run(), Table(), []
    with pytest.raises(ValueError, match="not confirmed"):
        approve(runs, decisions, lambda **kw: started.append(kw), "r1", 0.10, "haku", "ok",
                confirm=lambda summary: "yes")
    assert runs.items[("r1", "")]["status"] == "AWAITING_APPROVAL"
    assert not [k for k in decisions.items if k[1]] and not started


def test_without_a_terminal_the_approver_is_recorded_as_automation():
    # A name passed on the command line by a script is a claim, not a person.
    runs, decisions, started = halted_run(), Table(), []
    approve(runs, decisions, lambda **kw: started.append(kw), "r1", 0.10, "haku", "ci", confirm=None)
    rec = next(v for (r, d), v in decisions.items.items() if d)
    assert rec["input"]["approver"] == "automation"
    assert rec["input"]["approver_claimed"] == "haku"
    assert rec["input"]["channel"] == "non-interactive"
    assert started  # still allowed: the record says what it was


def test_the_cli_records_a_piped_approval_as_automation(monkeypatch):
    # CliRunner gives the command no terminal, as a script or an agent would.
    from typer.testing import CliRunner

    import blindspot.cloud.approve as mod
    from blindspot import cli

    seen = {}

    def fake_approve(runs, decisions, start, run_id, usd, approver, reason, confirm=None):
        seen["confirm"] = confirm
        return {"contract_version": 2, "budget_usd": 0.5}

    class Session:
        def __init__(self, **kw):
            pass

        def resource(self, name):
            return type("R", (), {"Table": lambda self, n: None})()

        def client(self, name):
            return type("C", (), {"describe_stacks": lambda self, **k: {"Stacks": [{"Outputs": []}]}})()

    monkeypatch.setattr(mod, "approve", fake_approve)
    monkeypatch.setattr("boto3.Session", Session)
    result = CliRunner().invoke(cli.app, ["approve", "--run-id", "r1", "--additional-usd", "0.1",
                                          "--approver", "haku", "--reason", "x"])
    assert result.exit_code == 0, result.output
    assert seen["confirm"] is None and "automation" in result.output
