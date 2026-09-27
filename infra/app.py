"""CDK entrypoint.  uv run --group infra cdk deploy --all  (see README)."""

import aws_cdk as cdk

from blindspot_stack import BlindspotStack

app = cdk.App()
BlindspotStack(
    app, "Blindspot",
    env=cdk.Environment(region=app.node.try_get_context("region") or "us-east-1"),
    synthesizer=cdk.DefaultStackSynthesizer(qualifier="bspot"),
    description="Blindspot: finds where a vision pipeline starts failing (project=blindspot)",
)
app.synth()
