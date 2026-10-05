"""List the AWS services the deployed stacks actually contain.

Reads the live CloudFormation resources of the Blindspot stack and its CDK
toolkit stack, groups them by service, and writes the result without account
IDs, ARNs or physical names. The Devpost "Built with" list comes from here, not
from memory.

    AWS_PROFILE=blindspot uv run --group cloud python tools/aws_services.py \
        --out bench/out/cloud_runs/aws_services.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import boto3

STACKS = ["Blindspot", "CDKToolkit-blindspot"]

#: CloudFormation namespace -> service name as AWS markets it.
NAMES = {
    "Batch": "AWS Batch", "CloudFront": "Amazon CloudFront", "CloudWatch": "Amazon CloudWatch",
    "DynamoDB": "Amazon DynamoDB", "EC2": "Amazon VPC", "ECR": "Amazon ECR",
    "ECS": "Amazon ECS (Fargate)", "Events": "Amazon EventBridge", "IAM": "AWS IAM",
    "Lambda": "AWS Lambda", "Logs": "Amazon CloudWatch Logs", "S3": "Amazon S3",
    "SSM": "AWS Systems Manager", "StepFunctions": "AWS Step Functions",
    "Budgets": "AWS Budgets", "KMS": "AWS KMS",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    cfn = boto3.client("cloudformation")
    by_service: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for stack in STACKS:
        for page in cfn.get_paginator("list_stack_resources").paginate(StackName=stack):
            for r in page["StackResourceSummaries"]:
                parts = r["ResourceType"].split("::")
                if parts[0] != "AWS" or parts[1] == "CDK" or r["ResourceStatus"].endswith("FAILED"):
                    continue  # Custom:: resources are Lambda-backed and counted there
                by_service[NAMES.get(parts[1], parts[1])][parts[2]] += 1
    # Batch on Fargate runs tasks through ECS without an ECS resource in the stack.
    if "AWS Batch" in by_service:
        by_service.setdefault("Amazon ECS (Fargate)", collections.Counter())
    # X-Ray has no resource of its own: it is on only if the state machine traces.
    sfn = boto3.client("stepfunctions")
    for m in sfn.list_state_machines()["stateMachines"]:
        if m["name"] == "bs-runs" and sfn.describe_state_machine(
                stateMachineArn=m["stateMachineArn"]).get("tracingConfiguration", {}).get("enabled"):
            by_service["AWS X-Ray"]["tracing (state machine)"] += 1
    result = {
        "stacks": STACKS,
        "services": {s: dict(sorted(c.items())) for s, c in sorted(by_service.items())},
        "service_count": len(by_service),
    }
    body = json.dumps(result, indent=1) + "\n"
    if args.out:
        args.out.write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
