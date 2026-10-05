"""One planning round of a cloud run.

A pure function of the run definition and the probe ledger. The ledger is the
only state: a planner Lambda can be killed and retried at any point and will
plan the same next wave. This module decides nothing about pass or fail
itself -- it applies `boundary.criterion` to measured mAP -- and it cannot
spend beyond the run's `cost.contract`: a wave the contract cannot afford is
cut to what it can, and a round that can afford nothing halts the run in
AWAITING_APPROVAL with the findings so far.

No AWS here; `plan_handler` is the thin adapter that reads and writes it.
"""

from __future__ import annotations

from ..axis import Axis
from ..boundary.criterion import Criterion
from ..boundary.planner import plan_axis
from ..cost.contract import BudgetContract

BASELINE_ID = "baseline"


def _axis(spec: dict) -> Axis:
    field = spec["axis"].partition(".")[2]
    return Axis(field, spec["unit"], spec["lo"], spec["hi"], severe_end=spec["severe_end"])


def _probe_id(axis_id: str, value: float) -> str:
    return f"{axis_id}@{float(value)!r}"


def _command(axis_id: str, value: float, seed: int) -> str:
    return f"uv run blindspot probe --set {axis_id}={float(value)!r} --seed {seed}"


def plan_round(run: dict, ledger: list[dict]) -> dict:
    contract = BudgetContract(run["budget_usd"], run["cost_per_probe_usd"])
    if run.get("prepaid_usd"):
        contract.charge_usd(run["prepaid_usd"], "agent model calls before the run")
    if ledger:
        contract.charge(len(ledger))  # already paid for

    by_id = {r["probe_id"]: r for r in ledger}
    if BASELINE_ID not in by_id:
        return _wave(contract, [{"probe_id": BASELINE_ID, "set": {}}], [])

    criterion = Criterion(baseline_map50=by_id[BASELINE_ID]["map50"])

    wanted, findings = [], []
    caps = run.get("probes_per_axis") or {}
    for spec in run["axes"]:
        axis = _axis(spec)
        observed = {}
        for r in ledger:
            if list(r["set"]) == [spec["axis"]]:
                observed[float(r["set"][spec["axis"]])] = criterion.failed(r["map50"])
        step = plan_axis(axis, observed, spec["target_width"],
                         verify_samples=run.get("verify_samples", 0))
        cap = caps.get(spec["axis"])
        if not step.done and cap is not None and len(observed) >= cap:
            # The accepted allocation is spent before the search converged:
            # unresolved, so no bracket and nothing to reproduce.
            findings.append({"axis": spec["axis"], "unit": spec["unit"], "status": "capped",
                             "lower": None, "upper": None, "probes_used": len(observed),
                             "reproduce": None})
            continue
        if step.done:
            fail_value = None
            if step.lower is not None:
                fail_value = step.upper if axis.severe_end == "hi" else step.lower
            findings.append({
                "axis": spec["axis"], "unit": spec["unit"], "status": step.status,
                "lower": step.lower, "upper": step.upper, "probes_used": step.probes_used,
                "reproduce": _command(spec["axis"], fail_value, run["seed"]) if fail_value is not None else None,
            })
        else:
            values = step.next_values if cap is None else step.next_values[: cap - len(observed)]
            wanted += [{"probe_id": _probe_id(spec["axis"], v), "set": {spec["axis"]: v}}
                       for v in values]

    if not wanted:
        return {"action": "done", "findings": findings, "criterion": criterion.describe(),
                **_money(contract)}
    return _wave(contract, wanted, findings)


def _money(contract: BudgetContract) -> dict:
    return {"spent_usd": round(contract.spent_usd, 6), "remaining_usd": round(contract.remaining_usd, 6)}


def _wave(contract: BudgetContract, wanted: list[dict], findings: list[dict]) -> dict:
    affordable = wanted[: contract.probes_remaining]
    if not affordable:
        decision = contract.request(len(wanted))
        return {"action": "halted", "state": decision.state.value, "reason": decision.reason,
                "requested_probes": len(wanted), "findings": findings, **_money(contract)}
    return {"action": "probe", "probes": affordable,
            "truncated": len(affordable) < len(wanted), **_money(contract)}
