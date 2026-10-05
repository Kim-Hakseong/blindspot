"""Human approval of a run halted on its budget contract.

The contract cannot be raised by the code operating under it, and an agent
has no tool that raises it. Continuing is a separate, human act: it must name
an approver and a reason, it adds a positive amount as a new contract version,
and it is written to the decision ledger before the loop restarts.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal

from ..cost.contract import BudgetContract


def approve(runs, decisions, start_execution, run_id: str, additional_usd: float,
            approver: str, reason: str) -> dict:
    if additional_usd <= 0:
        raise ValueError("approval must add a positive amount")
    if not approver.strip() or not reason.strip():
        raise ValueError("approval needs a named approver and a reason")
    item = runs.get_item(Key={"run_id": run_id}).get("Item")
    if item is None:
        raise ValueError(f"no run {run_id}")
    if item["status"] != "AWAITING_APPROVAL":
        raise ValueError(f"run {run_id} is {item['status']}, not AWAITING_APPROVAL")

    definition = json.loads(item["definition"])
    before = float(definition["budget_usd"])
    definition["budget_usd"] = round(before + additional_usd, 6)
    version = int(item.get("contract_version", 1)) + 1

    # An agent allocation that halted the run is applied once the new contract
    # can pay for it; otherwise it stays pending and the run continues without it.
    applied = None
    pending = definition.get("pending_allocation")
    if pending:
        contract = BudgetContract(definition["budget_usd"], definition["cost_per_probe_usd"])
        if definition.get("prepaid_usd"):
            contract.charge_usd(definition["prepaid_usd"], "agent model calls before the run")
        if contract.request(sum(pending.values())).approved:
            definition["probes_per_axis"] = definition.pop("pending_allocation")
            applied = pending

    decisions.put_item(Item=json.loads(json.dumps({
        "run_id": run_id, "decision_id": f"approve-v{version:02d}-{int(time.time())}",
        "tool": "human.approve",
        "input": {"additional_usd": additional_usd, "approver": approver},
        "output": {"budget_before_usd": before, "budget_after_usd": definition["budget_usd"],
                   "contract_version": version, "applied_allocation": applied},
        "rationale": reason, "accepted_by_scheduler": True,
    }), parse_float=Decimal))
    runs.put_item(Item={**item, "status": "RUNNING", "contract_version": version,
                        "definition": json.dumps(definition), "updated": int(time.time())})
    start_execution(run_id=run_id, arch=item.get("arch", "arm64"), contract_version=version)
    return {"run_id": run_id, "budget_usd": definition["budget_usd"], "contract_version": version}
