"""Run cost from billable resource usage (cost/ledger.py).

Pure arithmetic over usage records, so the number a run reports can be checked
by hand. Prices are the published us-east-1 on-demand rates the module states.
"""

import pytest

from blindspot.cost.ledger import (
    FARGATE, LAMBDA_ARM_GB_S, SFN_TRANSITION, FargateTask, run_cost,
)


def test_fargate_bills_vcpu_and_memory_per_second():
    t = FargateTask(arch="arm64", vcpu=2, memory_gb=4, seconds=120)
    price = FARGATE["arm64"]
    expected = 120 / 3600 * (2 * price["vcpu_h"] + 4 * price["gb_h"])
    assert run_cost([t])["fargate_usd"] == pytest.approx(expected)


def test_fargate_has_a_one_minute_minimum():
    short = FargateTask(arch="arm64", vcpu=2, memory_gb=4, seconds=5)
    minute = FargateTask(arch="arm64", vcpu=2, memory_gb=4, seconds=60)
    assert run_cost([short])["fargate_usd"] == pytest.approx(run_cost([minute])["fargate_usd"])


def test_x86_costs_more_than_graviton_for_the_same_task():
    arm = run_cost([FargateTask("arm64", 2, 4, 300)])["fargate_usd"]
    x86 = run_cost([FargateTask("x86", 2, 4, 300)])["fargate_usd"]
    assert x86 > arm


def test_lambda_and_step_functions_are_included():
    out = run_cost([], lambda_gb_seconds=10.0, sfn_transitions=40)
    assert out["lambda_usd"] == pytest.approx(10.0 * LAMBDA_ARM_GB_S)
    assert out["sfn_usd"] == pytest.approx(40 * SFN_TRANSITION)
    assert out["total_usd"] == pytest.approx(out["lambda_usd"] + out["sfn_usd"])


def test_report_states_what_it_does_not_count():
    out = run_cost([])
    assert "excluded" in out and out["excluded"]


def test_the_agents_model_cost_is_part_of_the_run_cost():
    # Run cost = Batch + Step Functions + Bedrock (definition of run cost).
    out = run_cost([], bedrock_usd=0.0126)
    assert out["bedrock_usd"] == pytest.approx(0.0126)
    assert out["total_usd"] == pytest.approx(0.0126)
