"""Cancel a run that must not resume, with the reason in the decision ledger.

Only a run that is not executing can be cancelled (AWAITING_APPROVAL or
PLANNING); a running loop is stopped by its contract, not by this. A
cancelled run cannot be approved afterwards. As with approvals, an operator
name typed without an interactive terminal is recorded as a claim, and the
operator as "automation".
"""

from __future__ import annotations

import json
import time
from decimal import Decimal

CANCELLABLE = {"AWAITING_APPROVAL", "PLANNING"}


def cancel(runs, decisions, run_id: str, reason: str, operator: str, interactive: bool) -> dict:
    if not reason.strip():
        raise ValueError("cancelling needs a reason")
    item = runs.get_item(Key={"run_id": run_id}).get("Item")
    if item is None:
        raise ValueError(f"no run {run_id}")
    if item["status"] not in CANCELLABLE:
        raise ValueError(f"run {run_id} is {item['status']}; only {sorted(CANCELLABLE)} can be cancelled")
    who = ({"operator": operator, "channel": "interactive"} if interactive else
           {"operator": "automation", "operator_claimed": operator, "channel": "non-interactive"})
    decisions.put_item(Item=json.loads(json.dumps({
        "run_id": run_id, "decision_id": f"cancel-{int(time.time())}", "tool": "operator.cancel",
        "input": {"status_before": item["status"], **who}, "output": {"status": "CANCELLED"},
        "rationale": reason, "accepted_by_scheduler": True,
    }), parse_float=Decimal))
    runs.put_item(Item={**item, "status": "CANCELLED", "cancel_reason": reason, "updated": int(time.time())})
    return {"run_id": run_id, "status": "CANCELLED"}
