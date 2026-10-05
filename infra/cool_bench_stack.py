"""Temporary EC2 stack for the three-arm COOL benchmark (rule K2).

COOL ships as a Graviton4 AMI, not a container, so it cannot run on the Fargate
workers. This stack launches two instances that each run
`bench/ec2/run_arms.sh` from a clone of the public repository at a pinned
commit, upload their reports to a private bucket, and terminate themselves:

  x86       c7i.large, stock Ubuntu 24.04  -> arm 1: stock OpenCV 5 wheel
  graviton  c8g.large, the COOL AMI        -> arm 2: stock wheel, arm 3: COOL
  (the Graviton size is selectable, and the x86 arm can be left out, for a
  rerun of arms 2 and 3 on a larger instance)

Arms 2 and 3 share one machine so the COOL effect is measured without a
hardware difference. Nothing is built locally. The stack exists only for the
benchmark: `bench/ec2_cool.py` deploys it, collects the reports and destroys
it. Deployed only when `-c cool_ami=<ami-id>` is given.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

REPO = "https://github.com/Kim-Hakseong/blindspot.git"
UV_VERSION = "0.11.12"
UBUNTU_AMD64 = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
LIFETIME_MINUTES = 60

BOOTSTRAP = """#!/bin/bash
shutdown -h +{lifetime}
set -uxo pipefail
exec > /var/log/blindspot-bench.log 2>&1
export HOME=/root AWS_DEFAULT_REGION=us-east-1
ROLE={role} BUCKET={bucket} PREFIX={prefix}
finish() {{
  rc=$?
  /opt/bs/.venv/bin/python - "$rc" "$BUCKET" "$PREFIX" "$ROLE" <<'EOF' || true
import sys, boto3
rc, bucket, prefix, role = sys.argv[1:]
s3 = boto3.client("s3")
s3.upload_file("/var/log/blindspot-bench.log", bucket, f"{{prefix}}/logs/{{role}}.log")
s3.put_object(Bucket=bucket, Key=f"{{prefix}}/done-{{role}}.json", Body=('{{"exit": %s}}' % rc).encode())
EOF
  shutdown -h now
}}
trap finish EXIT
if ! command -v shasum; then
  printf '#!/bin/sh\\n[ "$1" = "-a" ] && shift 2\\nexec sha256sum "$@"\\n' > /usr/local/bin/shasum
  chmod +x /usr/local/bin/shasum
fi
curl -LsSf https://astral.sh/uv/{uv}/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
git clone {repo} /opt/bs
cd /opt/bs
git checkout {commit}
bash bench/ec2/run_arms.sh "$ROLE" "$BUCKET" "$PREFIX" {digest}
"""


class CoolBenchStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, *, cool_ami: str, repo_commit: str,
                 manifest_digest: str, prefix: str = "bench", graviton_instance_type: str = "c8g.large",
                 include_x86: bool = True, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        cdk.Tags.of(self).add("project", "blindspot")

        # c8g and c7i are both offered in us-east-1a; one public subnet, no NAT.
        vpc = ec2.Vpc(self, "Vpc", availability_zones=["us-east-1a"], nat_gateways=0,
                      subnet_configuration=[ec2.SubnetConfiguration(
                          name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)])
        sg = ec2.SecurityGroup(self, "NoInbound", vpc=vpc, allow_all_outbound=True,
                               description="Benchmark instances: outbound only")
        results = s3.Bucket(self, "Results", block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                            enforce_ssl=True, encryption=s3.BucketEncryption.S3_MANAGED,
                            removal_policy=cdk.RemovalPolicy.DESTROY)
        role = iam.Role(self, "BenchRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))
        role.add_to_policy(iam.PolicyStatement(actions=["s3:PutObject"],
                                               resources=[results.arn_for_objects(f"{prefix}/*")]))

        arms = ((("x86", "c7i.large", ec2.MachineImage.from_ssm_parameter(UBUNTU_AMD64)),)
                if include_x86 else ()) + (
            ("graviton", graviton_instance_type, ec2.MachineImage.generic_linux({"us-east-1": cool_ami})),)
        for name, instance_type, image in arms:
            script = BOOTSTRAP.format(lifetime=LIFETIME_MINUTES, role=name, bucket=results.bucket_name,
                                      prefix=prefix, uv=UV_VERSION, repo=REPO, commit=repo_commit,
                                      digest=manifest_digest)
            instance = ec2.Instance(
                self, f"Bench_{name}", vpc=vpc, vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                instance_type=ec2.InstanceType(instance_type), machine_image=image,
                security_group=sg, role=role, user_data=ec2.UserData.custom(script),
                instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
                require_imdsv2=True, associate_public_ip_address=True,
                instance_name=f"bs-bench-{name}",
            )
            cdk.CfnOutput(self, f"InstanceId{name.capitalize()}", value=instance.instance_id)
        cdk.CfnOutput(self, "ResultsBucket", value=results.bucket_name)
