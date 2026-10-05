"""What an agent's proposals change in a cloud run, decided deterministically.

`blindspot cloud-run --agent` runs the agent before the Step Functions loop.
The gate (`cost.gate`) has already accepted or refused each proposal; this
module turns the outcome into the run definition and its starting status:

- an accepted axis order reorders the run's axes (it only matters when the
  contract truncates a wave: earlier axes are bought first);
- an accepted split becomes per-axis probe caps (`probes_per_axis`), which
  the planner honours by reporting a capped axis as unresolved;
- the agent's measured model cost becomes `prepaid_usd`, charged to the run's
  own contract before any probe;
- a split the contract cannot pay for -- refused by the gate, or unaffordable
  once the model cost is charged -- is stored as `pending_allocation`, and the
  run starts in AWAITING_APPROVAL. Resuming is the W4 human approval
  (`cloud.approve`), which applies the split if the new contract can pay.

No model client here: the agent runs in the CLI and hands over its outcome.
"""

from __future__ import annotations

import copy
import json
import time
from decimal import Decimal

from ..cost.contract import BudgetContract, BudgetExceeded


class DynamoDecisionLedger:
    """The agent's decision ledger, written to bs-decisions under one run id.

    Ids sort before the planner's rounds ("0001-plan"): the agent decides
    before round one.
    """

    def __init__(self, table, run_id: str):
        self.table, self.run_id, self.seq = table, run_id, 0

    def record(self, entry: dict) -> dict:
        self.seq += 1
        item = {"run_id": self.run_id, "decision_id": f"0000-agent-{self.seq:03d}",
                "at": int(time.time()), **entry}
        self.table.put_item(Item=json.loads(json.dumps(item, default=str), parse_float=Decimal))
        return item


def prepare_agent_run(definition: dict, outcome: dict) -> tuple[dict, str]:
    d = copy.deepcopy(definition)
    names = [a["axis"] for a in d["axes"]]
    order = outcome.get("axis_order") or names
    if sorted(order) == sorted(names) and len(set(order)) == len(order):
        by_name = {a["axis"]: a for a in d["axes"]}
        d["axes"] = [by_name[n] for n in order]

    d["prepaid_usd"] = round(float(outcome.get("model_usd", 0.0)), 6)
    contract = BudgetContract(d["budget_usd"], d["cost_per_probe_usd"])
    try:
        contract.charge_usd(d["prepaid_usd"], "agent model calls before the run")
    except BudgetExceeded:
        return d, "AWAITING_APPROVAL"

    pending = outcome.get("pending_allocation")
    allocation = outcome.get("allocation")
    if allocation and not contract.request(sum(allocation.values())).approved:
        pending, allocation = allocation, None
    if allocation:
        d["probes_per_axis"] = dict(allocation)
    if pending:
        d["pending_allocation"] = dict(pending)
        return d, "AWAITING_APPROVAL"
    return d, "RUNNING"
