"""Round-based planning for the cloud probe loop.

Step Functions runs in rounds: plan, fan out a wave of probes on Batch, plan
again. The planner is a pure function of the observations so far, and it does
not re-implement the search: it *replays* `locate_on_axis` against what has
been observed and stops at the first probe the search asks for that has not
been run. Two copies of the algorithm could drift apart; one copy replayed
cannot, which is what guarantees a cloud run and a local run of the same
condition visit the same probes and report the same boundary.

The verified mode's scan points do not depend on each other, so they are
emitted as one parallel wave -- computed with exactly the float expression the
search itself uses, so the values match bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .locate import locate_on_axis


@dataclass(frozen=True)
class PlanStep:
    done: bool
    next_values: tuple[float, ...] = ()
    status: str | None = None
    lower: float | None = None
    upper: float | None = None
    probes_used: int = 0


class _Need(Exception):
    def __init__(self, value: float):
        self.value = value


def _lookup(observed: dict[float, bool], value: float) -> bool | None:
    for known, failed in observed.items():
        if math.isclose(known, value, rel_tol=1e-12, abs_tol=1e-12):
            return failed
    return None


def scan_values(axis, verify_samples: int) -> list[float]:
    """The verified mode's scan points, by the search's own expression."""
    step = (1.0 - 0.0) / (verify_samples - 1)
    return [axis.from_severity(0.0 + i * step) for i in range(verify_samples)]


def plan_axis(
    axis,
    observed: dict[float, bool],
    target_width: float,
    verify_samples: int = 0,
    max_probes: int = 64,
) -> PlanStep:
    """Next probes for one axis, or the finished result."""
    if verify_samples > 1:
        missing = tuple(v for v in scan_values(axis, verify_samples) if _lookup(observed, v) is None)
        if missing:
            return PlanStep(done=False, next_values=missing)

    def evaluate(value: float) -> bool:
        known = _lookup(observed, value)
        if known is None:
            raise _Need(value)
        return known

    try:
        result = locate_on_axis(axis, evaluate, target_width, max_probes=max_probes,
                                verify_samples=verify_samples)
    except _Need as need:
        return PlanStep(done=False, next_values=(need.value,))
    return PlanStep(done=True, status=result.status.value, lower=result.lower,
                    upper=result.upper, probes_used=result.probes_used)
