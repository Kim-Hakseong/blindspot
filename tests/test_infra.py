"""The CDK stack, checked at synth time (W4-2 .. W4-6).

These assertions are the enforcement for the cloud rules, so they run on every
test pass without an AWS account: every resource tagged project=blindspot (the
account hosts another project), no NAT gateway or interface endpoint (the
fixed monthly cost), no IAM statement on Resource "*" beyond the documented
AWS-imposed exceptions, a public-access block on the bucket, and a vCPU cap on
every Batch compute environment.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

from blindspot_stack import BlindspotStack, MAX_VCPUS  # noqa: E402

#: Actions AWS only authorises on Resource "*". Each is listed with why.
STAR_ALLOWED = {
    "xray:PutTraceSegments": "X-Ray does not support resource-level permissions",
    "xray:PutTelemetryRecords": "X-Ray does not support resource-level permissions",
    "xray:GetSamplingRules": "X-Ray does not support resource-level permissions",
    "xray:GetSamplingTargets": "X-Ray does not support resource-level permissions",
    "logs:CreateLogDelivery": "Step Functions log delivery requires Resource *",
    "logs:GetLogDelivery": "Step Functions log delivery requires Resource *",
    "logs:UpdateLogDelivery": "Step Functions log delivery requires Resource *",
    "logs:DeleteLogDelivery": "Step Functions log delivery requires Resource *",
    "logs:ListLogDeliveries": "Step Functions log delivery requires Resource *",
    "logs:PutResourcePolicy": "Step Functions log delivery requires Resource *",
    "logs:DescribeResourcePolicies": "Step Functions log delivery requires Resource *",
    "logs:DescribeLogGroups": "Step Functions log delivery requires Resource *",
    "ecr:GetAuthorizationToken": "ECR login token is account-wide by design (image assets only)",
}

#: Allowed only when real image assets are synthesised, so absent in these tests.
ASSET_ONLY = {"ecr:GetAuthorizationToken"}


@pytest.fixture(scope="module")
def template():
    app = cdk.App(context={"budget_email": "alerts@example.invalid", "skip_docker": "1",
                           "skip_viewer": "1"})
    stack = BlindspotStack(app, "Blindspot",
                           env=cdk.Environment(account="123456789012", region="us-east-1"))
    return assertions.Template.from_stack(stack)


def resources(template, type_=None):
    items = template.to_json()["Resources"].items()
    return {k: v for k, v in items if type_ is None or v["Type"] == type_}


TAGGABLE = {
    "AWS::S3::Bucket", "AWS::DynamoDB::Table", "AWS::EC2::VPC", "AWS::EC2::Subnet",
    "AWS::Batch::ComputeEnvironment", "AWS::Batch::JobQueue", "AWS::Batch::JobDefinition",
    "AWS::StepFunctions::StateMachine", "AWS::Lambda::Function", "AWS::IAM::Role",
    "AWS::Logs::LogGroup",
    "AWS::CloudFront::Distribution",
}


def test_every_taggable_resource_carries_the_project_tag(template):
    untagged = []
    for name, res in resources(template).items():
        if res["Type"] not in TAGGABLE:
            continue
        tags = res.get("Properties", {}).get("Tags", [])
        if isinstance(tags, dict):  # Batch uses a map
            ok = tags.get("project") == "blindspot"
        else:
            ok = {"Key": "project", "Value": "blindspot"} in tags
        if not ok:
            untagged.append(f"{name} ({res['Type']})")
    assert untagged == []


def test_no_nat_gateway_or_interface_endpoint(template):
    assert resources(template, "AWS::EC2::NatGateway") == {}
    endpoints = resources(template, "AWS::EC2::VPCEndpoint")
    assert all(e["Properties"].get("VpcEndpointType") == "Gateway" for e in endpoints.values())


def test_no_iam_statement_on_star_except_documented(template):
    offenders = []
    for kind in ("AWS::IAM::Policy", "AWS::IAM::Role"):
        for name, res in resources(template, kind).items():
            docs = [res["Properties"].get("PolicyDocument")] + [
                p["PolicyDocument"] for p in res["Properties"].get("Policies", [])
            ]
            for doc in filter(None, docs):
                for st in doc["Statement"]:
                    res_ = st.get("Resource")
                    if res_ == "*" or (isinstance(res_, list) and "*" in res_):
                        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
                        for a in actions:
                            if a not in STAR_ALLOWED:
                                offenders.append(f"{name}: {a}")
    assert offenders == [], "\n".join(offenders)


def test_star_allowlist_has_no_unused_entries(template):
    """An allowlist that grows speculatively stops meaning anything."""
    used = set()
    for kind in ("AWS::IAM::Policy", "AWS::IAM::Role"):
        for res in resources(template, kind).values():
            docs = [res["Properties"].get("PolicyDocument")] + [
                p["PolicyDocument"] for p in res["Properties"].get("Policies", [])]
            for doc in filter(None, docs):
                for st in doc["Statement"]:
                    r = st.get("Resource")
                    if r == "*" or (isinstance(r, list) and "*" in r):
                        used.update(st["Action"] if isinstance(st["Action"], list) else [st["Action"]])
    assert set(STAR_ALLOWED) - ASSET_ONLY - used == set()


def test_bucket_blocks_public_access_and_requires_tls(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "PublicAccessBlockConfiguration": {
            "BlockPublicAcls": True, "BlockPublicPolicy": True,
            "IgnorePublicAcls": True, "RestrictPublicBuckets": True,
        }
    })
    policies = json.dumps(resources(template, "AWS::S3::BucketPolicy"))
    assert "aws:SecureTransport" in policies


def test_every_compute_environment_is_capped(template):
    envs = resources(template, "AWS::Batch::ComputeEnvironment")
    assert len(envs) == 2
    for env in envs.values():
        assert env["Properties"]["ComputeResources"]["MaxvCpus"] <= MAX_VCPUS


def test_both_architectures_have_a_job_definition(template):
    defs = resources(template, "AWS::Batch::JobDefinition")
    archs = sorted(d["Properties"]["ContainerProperties"]["RuntimePlatform"]["CpuArchitecture"]
                   for d in defs.values())
    assert archs == ["ARM64", "X86_64"]


def test_ledger_tables_exist_on_demand(template):
    tables = resources(template, "AWS::DynamoDB::Table")
    names = sorted(t["Properties"]["TableName"] for t in tables.values())
    assert names == ["bs-decisions", "bs-probes", "bs-runs"]
    for t in tables.values():
        assert t["Properties"]["BillingMode"] == "PAY_PER_REQUEST"


def test_logs_expire(template):
    for g in resources(template, "AWS::Logs::LogGroup").values():
        assert g["Properties"].get("RetentionInDays", 0) <= 14


def test_budget_alarm_is_defined(template):
    budgets = resources(template, "AWS::Budgets::Budget")
    assert len(budgets) == 1
    b = next(iter(budgets.values()))["Properties"]
    assert b["Budget"]["BudgetLimit"]["Amount"] == 25
    filt = json.dumps(b["Budget"].get("CostFilters", {}) or b["Budget"].get("FilterExpression", {}))
    assert "project" in filt and "blindspot" in filt


def test_state_machine_has_plan_submit_and_halt(template):
    sm = next(iter(resources(template, "AWS::StepFunctions::StateMachine").values()))
    definition = json.dumps(sm["Properties"]["DefinitionString"])
    for state in ("Plan", "SubmitWave", "Finalize", "Halt"):
        assert state in definition


def test_dashboard_shows_planner_metrics_and_run_outcomes(template):
    dashboards = resources(template, "AWS::CloudWatch::Dashboard")
    assert len(dashboards) == 1
    body = json.dumps(next(iter(dashboards.values()))["Properties"]["DashboardBody"])
    for metric in ("ProbesCompleted", "SpentUSD", "AxesLocated", "RunHalted", "ExecutionsFailed"):
        assert metric in body



def test_report_viewer_is_served_by_cloudfront_from_a_private_bucket(template):
    """W5-7: public URL, no credentials, no server -- static S3 behind CloudFront.
    The bucket itself stays private; only this distribution may read it."""
    dists = resources(template, "AWS::CloudFront::Distribution")
    assert len(dists) == 1
    cfg = next(iter(dists.values()))["Properties"]["DistributionConfig"]
    assert cfg["DefaultRootObject"] == "index.html"
    assert cfg["DefaultCacheBehavior"]["ViewerProtocolPolicy"] == "redirect-to-https"
    assert "OriginAccessControlId" in json.dumps(cfg["Origins"])
    assert len(resources(template, "AWS::CloudFront::OriginAccessControl")) == 1
    buckets = resources(template, "AWS::S3::Bucket")
    for b in buckets.values():
        assert b["Properties"]["PublicAccessBlockConfiguration"]["RestrictPublicBuckets"] is True
    policies = json.dumps(resources(template, "AWS::S3::BucketPolicy"))
    assert "cloudfront.amazonaws.com" in policies and "AWS:SourceArn" in policies


def test_viewer_url_is_an_output(template):
    outputs = template.to_json()["Outputs"]
    assert any(k.startswith("ReportUrl") for k in outputs)


def test_published_worker_images_can_be_reused_without_a_local_build():
    # `-c worker_tag_arm64=<tag> -c worker_tag_x86=<tag>` points the job
    # definitions at images already in the CDK asset repository, so a change
    # that does not touch the worker redeploys without a Docker build.
    app = cdk.App(context={"skip_viewer": "1", "worker_tag_arm64": "a" * 64,
                           "worker_tag_x86": "b" * 64})
    stack = BlindspotStack(app, "Blindspot",
                           env=cdk.Environment(account="123456789012", region="us-east-1"))
    t = assertions.Template.from_stack(stack).to_json()
    images = {jd["Properties"]["JobDefinitionName"]: json.dumps(jd["Properties"])
              for jd in t["Resources"].values() if jd["Type"] == "AWS::Batch::JobDefinition"}
    assert "cdk-bspot-container-assets" in images["bs-worker-arm64"] and "a" * 64 in images["bs-worker-arm64"]
    assert "b" * 64 in images["bs-worker-x86"]
    assembly = app.synth()
    manifest = json.loads(pathlib.Path(assembly.directory, "Blindspot.assets.json").read_text())
    assert not manifest.get("dockerImages")
