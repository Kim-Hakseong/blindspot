"""CDK entrypoint.  uv run --group infra cdk deploy --all  (see README)."""

import aws_cdk as cdk

from blindspot_stack import BlindspotStack
from cool_bench_stack import CoolBenchStack

app = cdk.App()
BlindspotStack(
    app, "Blindspot",
    env=cdk.Environment(region=app.node.try_get_context("region") or "us-east-1"),
    synthesizer=cdk.DefaultStackSynthesizer(qualifier="bspot"),
    description="Blindspot: finds where a vision pipeline starts failing (project=blindspot)",
)
# Temporary three-arm COOL benchmark; bench/ec2_cool.py deploys and destroys it.
if app.node.try_get_context("cool_ami"):
    CoolBenchStack(
        app, "BlindspotCoolBench",
        cool_ami=app.node.try_get_context("cool_ami"),
        repo_commit=app.node.try_get_context("repo_commit"),
        manifest_digest=app.node.try_get_context("manifest_digest") or "f28484951393",
        env=cdk.Environment(region="us-east-1"),
        synthesizer=cdk.DefaultStackSynthesizer(qualifier="bspot"),
        description="Blindspot: temporary COOL benchmark instances (project=blindspot)",
    )
app.synth()
