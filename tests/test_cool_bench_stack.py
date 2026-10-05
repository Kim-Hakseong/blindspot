"""The temporary EC2 stack for the three-arm COOL benchmark (W5-2/3).

It launches paid instances from a Marketplace AMI, so the guarantees that keep
it cheap are tested at synth time: each instance terminates itself on shutdown,
schedules that shutdown for 60 minutes after boot before doing anything else,
uploads nothing anywhere but its own private bucket, and is tagged.
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

from cool_bench_stack import CoolBenchStack  # noqa: E402


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = CoolBenchStack(app, "BlindspotCoolBench", cool_ami="ami-0123456789abcdef0",
                           repo_commit="0" * 40, manifest_digest="f28484951393",
                           env=cdk.Environment(account="123456789012", region="us-east-1"))
    return assertions.Template.from_stack(stack)


def of_type(template, type_):
    return {k: v for k, v in template.to_json()["Resources"].items() if v["Type"] == type_}


def user_data(instance) -> str:
    return json.dumps(instance["Properties"]["UserData"])


def test_two_instances_one_per_chip(template):
    types = sorted(i["Properties"]["InstanceType"] for i in of_type(template, "AWS::EC2::Instance").values())
    assert types == ["c7i.large", "c8g.large"]


def test_graviton_instance_boots_the_cool_ami(template):
    graviton = [i for i in of_type(template, "AWS::EC2::Instance").values()
                if i["Properties"]["InstanceType"] == "c8g.large"][0]
    assert graviton["Properties"]["ImageId"] == "ami-0123456789abcdef0"


def test_every_instance_terminates_itself_within_60_minutes(template):
    for instance in of_type(template, "AWS::EC2::Instance").values():
        assert instance["Properties"]["InstanceInitiatedShutdownBehavior"] == "terminate"
        script = user_data(instance)
        # the 60-minute shutdown is scheduled before any step that could fail
        assert script.index("shutdown -h +60") < script.index("curl")
        assert "trap finish EXIT" in script and "shutdown -h now" in script


def test_benchmark_code_is_pinned_to_a_commit(template):
    for instance in of_type(template, "AWS::EC2::Instance").values():
        assert "git checkout " + "0" * 40 in user_data(instance)


def test_everything_is_tagged(template):
    for logical_id, r in template.to_json()["Resources"].items():
        if r["Type"] in {"AWS::EC2::Instance", "AWS::EC2::VPC", "AWS::S3::Bucket",
                         "AWS::IAM::Role", "AWS::EC2::SecurityGroup", "AWS::EC2::Subnet"}:
            tags = {t["Key"]: t["Value"] for t in r["Properties"].get("Tags", [])}
            assert tags.get("project") == "blindspot", logical_id


def test_no_inbound_access_no_nat_imdsv2(template):
    for sg in of_type(template, "AWS::EC2::SecurityGroup").values():
        assert not sg["Properties"].get("SecurityGroupIngress")
    assert not of_type(template, "AWS::EC2::NatGateway")
    for lt in of_type(template, "AWS::EC2::LaunchTemplate").values():
        assert lt["Properties"]["LaunchTemplateData"]["MetadataOptions"]["HttpTokens"] == "required"


def test_iam_is_scoped_to_the_results_bucket(template):
    for policy in of_type(template, "AWS::IAM::Policy").values():
        for st in policy["Properties"]["PolicyDocument"]["Statement"]:
            res = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            assert "*" not in res
            assert all(a.startswith("s3:") for a in
                       (st["Action"] if isinstance(st["Action"], list) else [st["Action"]]))


def test_results_bucket_is_private(template):
    for b in of_type(template, "AWS::S3::Bucket").values():
        cfg = b["Properties"]["PublicAccessBlockConfiguration"]
        assert all(cfg[k] for k in ("BlockPublicAcls", "BlockPublicPolicy",
                                    "IgnorePublicAcls", "RestrictPublicBuckets"))
