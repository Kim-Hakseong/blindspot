"""Boundary location.

Tested against synthetic oracles whose true boundary is known exactly, so a
localisation error is unambiguous and the tests run without a model or a
dataset.

The cases that matter most are the dishonest ones: a range where everything
fails, a range where nothing does, and a response with more than one
transition. Bisection assumes monotonicity, and a search that quietly returns
a confident answer on a non-monotone response is worse than one that refuses.
"""

from __future__ import annotations

import pytest

from blindspot.boundary.locate import (
    BoundaryStatus,
    bisect_boundary,
    locate_on_axis,
    scan_boundaries,
)


def step_at(threshold: float):
    """Oracle: passes below `threshold`, fails at or above it."""
    calls = {"n": 0}

    def evaluate(value: float) -> bool:
        calls["n"] += 1
        return value >= threshold  # True == failed

    evaluate.calls = calls
    return evaluate


# ------------------------------------------------------------- bisection


def test_bisection_brackets_the_true_boundary():
    result = bisect_boundary(step_at(7.0), lo=0.0, hi=20.0, target_width=0.5)
    assert result.status is BoundaryStatus.LOCATED
    assert result.lower <= 7.0 <= result.upper
    assert result.width <= 0.5


def test_bisection_width_shrinks_to_the_target():
    for target in (2.0, 0.5, 0.1):
        result = bisect_boundary(step_at(13.3), lo=0.0, hi=40.0, target_width=target)
        assert result.width <= target
        assert result.lower <= 13.3 <= result.upper


def test_bisection_probe_count_is_logarithmic():
    """40 wide down to 0.5 needs about log2(80) ~ 6.3 -> 7 bisections, plus the
    two endpoint probes."""
    result = bisect_boundary(step_at(13.3), lo=0.0, hi=40.0, target_width=0.5)
    assert result.probes_used <= 10, f"used {result.probes_used} probes"
    assert result.probes_used >= 3


def test_bisection_reports_when_everything_fails():
    """A range whose benign end already fails has no boundary inside it."""
    result = bisect_boundary(step_at(-1.0), lo=0.0, hi=20.0, target_width=0.5)
    assert result.status is BoundaryStatus.FAILS_THROUGHOUT
    assert result.lower is None and result.upper is None


def test_bisection_reports_when_nothing_fails():
    result = bisect_boundary(step_at(1000.0), lo=0.0, hi=20.0, target_width=0.5)
    assert result.status is BoundaryStatus.PASSES_THROUGHOUT
    assert result.lower is None and result.upper is None


def test_bisection_does_not_probe_beyond_its_budget():
    result = bisect_boundary(step_at(7.0), lo=0.0, hi=20.0, target_width=1e-9, max_probes=6)
    assert result.probes_used <= 6
    assert result.status is BoundaryStatus.BUDGET_EXHAUSTED


def test_bisection_records_every_probe_it_made():
    result = bisect_boundary(step_at(7.0), lo=0.0, hi=20.0, target_width=0.5)
    assert len(result.probes) == result.probes_used
    for value, failed in result.probes:
        assert failed == (value >= 7.0)


def test_bisection_is_deterministic():
    a = bisect_boundary(step_at(7.0), lo=0.0, hi=20.0, target_width=0.25)
    b = bisect_boundary(step_at(7.0), lo=0.0, hi=20.0, target_width=0.25)
    assert (a.lower, a.upper, a.probes_used) == (b.lower, b.upper, b.probes_used)


def test_bisection_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        bisect_boundary(step_at(7.0), lo=20.0, hi=0.0, target_width=0.5)


# ------------------------------------------------- non-monotone responses


def test_scan_finds_a_single_transition():
    samples = [(0.0, False), (5.0, False), (10.0, True), (15.0, True)]
    found = scan_boundaries(samples)
    assert len(found) == 1
    assert found[0] == (5.0, 10.0)


def test_scan_finds_multiple_transitions():
    """A response that fails, recovers, then fails again has three transitions.

    Both directions count: pass->fail bounds a failure region and fail->pass
    bounds its far side. Reporting only pass->fail would describe the response
    as monotone when it is not, and the recovery would never be reported.
    """
    samples = [
        (0.0, False),
        (5.0, True),
        (10.0, False),
        (15.0, True),
    ]
    found = scan_boundaries(samples)
    assert found == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def test_scan_returns_nothing_when_there_is_no_transition():
    assert scan_boundaries([(0.0, False), (5.0, False)]) == []
    assert scan_boundaries([(0.0, True), (5.0, True)]) == []


def test_scan_sorts_unordered_samples():
    samples = [(10.0, True), (0.0, False), (5.0, False)]
    assert scan_boundaries(samples) == [(5.0, 10.0)]


def test_pure_bisection_misses_an_interior_failure_band():
    """A documented limitation, pinned so it cannot regress into a silent bug.

    With a failure band strictly inside the range, both endpoints pass and
    bisection concludes "no boundary" after two probes. This is why the
    verification sweep exists and why it scans before it bisects. The cheap
    mode is genuinely cheaper and genuinely blinder, and callers must be able
    to tell which one produced a result.
    """
    def evaluate(value: float) -> bool:
        return 5.0 <= value < 10.0

    cheap = bisect_boundary(evaluate, lo=0.0, hi=20.0, target_width=0.5)
    assert cheap.status is BoundaryStatus.PASSES_THROUGHOUT
    assert cheap.probes_used == 2

    verified = bisect_boundary(
        evaluate, lo=0.0, hi=20.0, target_width=0.5, verify_samples=9
    )
    assert verified.status is BoundaryStatus.NOT_MONOTONE


def test_bisection_flags_a_non_monotone_response():
    """With a verification sweep, a multi-transition response must not be
    reported as a single located boundary."""
    # Fails in the middle band only: 5 <= x < 10.
    def evaluate(value: float) -> bool:
        return 5.0 <= value < 10.0

    result = bisect_boundary(
        evaluate, lo=0.0, hi=20.0, target_width=0.5, verify_samples=9
    )
    assert result.status is BoundaryStatus.NOT_MONOTONE
    assert len(result.transitions) >= 2


# ------------------------------------------------- severity orientation


class FakeAxis:
    """Minimal stand-in for degrade.Axis, to test orientation in isolation."""

    def __init__(self, lo, hi, severe_end):
        self.lo, self.hi, self.severe_end = lo, hi, severe_end

    @property
    def benign(self):
        return self.lo if self.severe_end == "hi" else self.hi

    @property
    def severe(self):
        return self.hi if self.severe_end == "hi" else self.lo

    def from_severity(self, s):
        return self.benign + s * (self.severe - self.benign)


def test_locate_handles_an_axis_whose_low_end_is_severe():
    """Illuminance: 0.5 lux is harsh, 400 lux is benign.

    Searching in raw value order would probe 0.5 first, find it failing, and
    report 'fails throughout' on an axis that has a real boundary in it.
    """
    axis = FakeAxis(lo=0.5, hi=400.0, severe_end="lo")

    def evaluate(lux: float) -> bool:
        return lux < 20.0  # fails in the dark

    result = locate_on_axis(axis, evaluate, target_width=2.0)
    assert result.status is BoundaryStatus.LOCATED
    assert result.lower <= 20.0 <= result.upper
    assert result.upper - result.lower <= 2.0


def test_locate_handles_an_axis_whose_high_end_is_severe():
    axis = FakeAxis(lo=0.0, hi=40.0, severe_end="hi")

    def evaluate(ms: float) -> bool:
        return ms >= 11.5

    result = locate_on_axis(axis, evaluate, target_width=1.0)
    assert result.status is BoundaryStatus.LOCATED
    assert result.lower <= 11.5 <= result.upper


def test_locate_reports_bounds_in_ascending_axis_units():
    """Even on an inverted axis, lower must not exceed upper."""
    axis = FakeAxis(lo=5.0, hi=100.0, severe_end="lo")
    result = locate_on_axis(axis, lambda q: q < 12.0, target_width=1.0)
    assert result.lower < result.upper


def test_locate_probes_are_reported_in_axis_units():
    axis = FakeAxis(lo=0.5, hi=400.0, severe_end="lo")
    result = locate_on_axis(axis, lambda lux: lux < 20.0, target_width=5.0)
    for value, _ in result.probes:
        assert 0.5 <= value <= 400.0
