"""Run cost from billable usage. Pure arithmetic -- no AWS calls here.

`tools/cost_report.py` collects the usage of one run (Fargate task time from
ECS, Lambda duration, Step Functions transitions) and this module prices it.
Keeping the arithmetic separate means the cost a run reports can be checked by
hand, and the judgment package stays free of cloud clients.

Prices: AWS on-demand, us-east-1, as published in 2026; edit here if they
change. Fargate bills per second from image pull start to task stop, with a
one-minute minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

FARGATE = {
    "arm64": {"vcpu_h": 0.03238, "gb_h": 0.00356},
    "x86": {"vcpu_h": 0.04048, "gb_h": 0.004445},
}
LAMBDA_ARM_GB_S = 0.0000133334
LAMBDA_REQUEST = 0.0000002
SFN_TRANSITION = 0.000025
MIN_BILLED_SECONDS = 60


@dataclass(frozen=True)
class FargateTask:
    arch: str
    vcpu: float
    memory_gb: float
    seconds: float


def run_cost(tasks: list[FargateTask], lambda_gb_seconds: float = 0.0,
             lambda_requests: int = 0, sfn_transitions: int = 0) -> dict:
    fargate = 0.0
    billed_seconds = 0.0
    for t in tasks:
        seconds = max(t.seconds, MIN_BILLED_SECONDS)
        billed_seconds += seconds
        price = FARGATE[t.arch]
        fargate += seconds / 3600.0 * (t.vcpu * price["vcpu_h"] + t.memory_gb * price["gb_h"])
    lam = lambda_gb_seconds * LAMBDA_ARM_GB_S + lambda_requests * LAMBDA_REQUEST
    sfn = sfn_transitions * SFN_TRANSITION
    return {
        "fargate_usd": fargate, "fargate_tasks": len(tasks), "fargate_billed_seconds": billed_seconds,
        "lambda_usd": lam, "sfn_usd": sfn, "total_usd": fargate + lam + sfn,
        "excluded": ["DynamoDB on-demand requests (fractions of a cent per run)",
                     "S3 requests and storage", "CloudWatch Logs ingestion",
                     "public IPv4 address-hours during tasks", "ECR storage (monthly, not per run)"],
    }
