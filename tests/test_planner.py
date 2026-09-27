"""Round-based planning for the cloud probe loop (W4-4).

Step Functions runs in rounds: plan, fan out a wave of probes, plan again. The
planner is a pure function of what has been observed so far. The property
that matters is equivalence: for any response, replaying the planner round by
round must visit exactly the probes `locate_on_axis` visits locally and reach
the same boundary -- otherwise a cloud run and a local run of the same
condition would disagree.
"""

from __future__ import annotations

import pytest

from blindspot.boundary.locate import BoundaryStatus, locate_on_axis
from blindspot.boundary.planner import plan_axis
from blindspot.degrade import REGISTRY


def replay(axis, evaluate, target_width, verify_samples):
    observed: dict[float, bool] = {}
    rounds = 0
    while True:
        step = plan_axis(axis, observed, target_width, verify_samples=verify_samples)
        if step.done:
            return step, observed, rounds
        rounds += 1
        assert step.next_values, "a plan that is not done must request probes"
        for v in step.next_values:
            assert v not in observed, "planner re-requested an observed value"
            observed[v] = evaluate(v)


def oracle(threshold, severe_high=True):
    return (lambda v: v >= threshold) if severe_high else (lambda v: v <= threshold)


CASES = [
    ("motion_blur", "exposure_ms", 13.1, True),
    ("low_light", "illuminance_lux", 19.0, False),
    ("fog", "beta_per_m", 0.061, True),
    ("jpeg", "quality", 9.4, False),
]


@pytest.mark.parametrize("verify", [0, 5])
@pytest.mark.parametrize("deg,field,thr,high", CASES)
def test_planner_matches_local_search_exactly(deg, field, thr, high, verify):
    axis = REGISTRY.get(deg).axis(field)
    width = (axis.hi - axis.lo) / 32
    evaluate = oracle(thr, high)

    local_calls = []
    local = locate_on_axis(axis, lambda v: (local_calls.append(v), evaluate(v))[1],
                           target_width=width, verify_samples=verify)
    step, observed, _ = replay(axis, evaluate, width, verify)

    assert step.status == local.status.value
    assert (step.lower, step.upper) == pytest.approx((local.lower, local.upper))
    assert sorted(observed) == pytest.approx(sorted(set(local_calls)))


def test_scan_phase_is_one_parallel_round():
    axis = REGISTRY.get("motion_blur").axis("exposure_ms")
    first = plan_axis(axis, {}, (axis.hi - axis.lo) / 32, verify_samples=5)
    assert len(first.next_values) == 5


def test_bisection_asks_for_one_probe_per_round():
    axis = REGISTRY.get("motion_blur").axis("exposure_ms")
    evaluate = oracle(13.1)
    observed = {v: evaluate(v) for v in plan_axis(axis, {}, 1.25, verify_samples=5).next_values}
    nxt = plan_axis(axis, observed, 1.25, verify_samples=5)
    assert len(nxt.next_values) == 1


def test_round_count_is_small():
    """Wall time on Batch is dominated by rounds, not probes."""
    axis = REGISTRY.get("jpeg").axis("quality")
    _, _, rounds = replay(axis, oracle(9.4, False), (axis.hi - axis.lo) / 32, 5)
    assert rounds <= 6


def test_passes_throughout_is_reported():
    axis = REGISTRY.get("fog").axis("beta_per_m")
    step, _, _ = replay(axis, lambda v: False, (axis.hi - axis.lo) / 32, 5)
    assert step.status == BoundaryStatus.PASSES_THROUGHOUT.value


def test_plan_is_a_pure_function_of_observations():
    axis = REGISTRY.get("fog").axis("beta_per_m")
    obs = {0.0: False, 0.12: True}
    a = plan_axis(axis, dict(obs), 0.004, verify_samples=0)
    b = plan_axis(axis, dict(obs), 0.004, verify_samples=0)
    assert a == b
