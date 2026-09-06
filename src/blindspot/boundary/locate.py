"""Locating a failure boundary with as few probes as possible.

A probe is the unit of cost, and an exhaustive grid spends most of its budget
confirming what is already obvious -- that the benign end passes and the severe
end fails. Bisection spends it only where the answer is still in doubt, which
is why the boundary is reported as an *interval* rather than a point: the
interval is exactly the residual uncertainty the budget bought.

Bisection assumes the response is monotone along the axis. Detectors are not
guaranteed to oblige. So a located boundary can be checked with a verification
sweep, and a response with more than one transition is reported as
``NOT_MONOTONE`` rather than as a confident answer that happens to be wrong.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

#: `(axis value) -> did it fail?`
Evaluator = Callable[[float], bool]


class BoundaryStatus(enum.Enum):
    LOCATED = "located"
    PASSES_THROUGHOUT = "passes_throughout"
    FAILS_THROUGHOUT = "fails_throughout"
    NOT_MONOTONE = "not_monotone"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class BoundaryResult:
    """Where the pass -> fail transition is, and what it cost to find out."""

    status: BoundaryStatus
    lower: float | None = None
    upper: float | None = None
    probes_used: int = 0
    probes: list[tuple[float, bool]] = field(default_factory=list)
    transitions: list[tuple[float, float]] = field(default_factory=list)

    @property
    def width(self) -> float:
        """Residual uncertainty in the boundary position, in axis units."""
        if self.lower is None or self.upper is None:
            return float("inf")
        return self.upper - self.lower

    @property
    def midpoint(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return (self.lower + self.upper) / 2.0

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width if self.lower is not None else None,
            "midpoint": self.midpoint,
            "probes_used": self.probes_used,
            "transitions": self.transitions,
        }


def scan_boundaries(samples: Sequence[tuple[float, bool]]) -> list[tuple[float, float]]:
    """Every pass -> fail *and* fail -> pass transition in a sampled response.

    Used to tell a genuinely monotone response from one that merely looked
    monotone at the points bisection happened to visit.
    """
    ordered = sorted(samples, key=lambda s: s[0])
    transitions: list[tuple[float, float]] = []
    for (v0, f0), (v1, f1) in zip(ordered, ordered[1:]):
        if f0 != f1:
            transitions.append((v0, v1))
    return transitions


def locate_on_axis(
    axis,
    evaluate_value: Evaluator,
    target_width: float,
    max_probes: int = 64,
    verify_samples: int = 0,
) -> BoundaryResult:
    """Locate a boundary on an ``Axis``, respecting which end is severe.

    Bisection needs to walk from benign to severe. Half the physical axes run
    the other way -- more light and higher JPEG quality are *better* -- so
    searching in raw value order would start at the harsh end, find it already
    failing, and report "fails throughout" on an axis that has a perfectly good
    boundary in it. Severity is normalised to [0, 1] here and the result is
    mapped back into the axis's own unit, so callers cannot get it wrong.
    """
    severity_width = abs(target_width / (axis.severe - axis.benign))

    def evaluate_severity(severity: float) -> bool:
        return evaluate_value(axis.from_severity(severity))

    result = bisect_boundary(
        evaluate_severity,
        lo=0.0,
        hi=1.0,
        target_width=severity_width,
        max_probes=max_probes,
        verify_samples=verify_samples,
    )

    def to_value(severity):
        return None if severity is None else axis.from_severity(severity)

    lower_value = to_value(result.lower)
    upper_value = to_value(result.upper)
    # from_severity may invert the ordering; report lower <= upper in axis units.
    if lower_value is not None and upper_value is not None and lower_value > upper_value:
        lower_value, upper_value = upper_value, lower_value

    return BoundaryResult(
        status=result.status,
        lower=lower_value,
        upper=upper_value,
        probes_used=result.probes_used,
        probes=[(axis.from_severity(s), failed) for s, failed in result.probes],
        transitions=[
            tuple(sorted((axis.from_severity(a), axis.from_severity(b))))
            for a, b in result.transitions
        ],
    )


def bisect_boundary(
    evaluate: Evaluator,
    lo: float,
    hi: float,
    target_width: float,
    max_probes: int = 64,
    verify_samples: int = 0,
) -> BoundaryResult:
    """Bisect for the pass -> fail transition between ``lo`` and ``hi``.

    ``lo`` is the benign end. Both endpoints are probed first: if the benign end
    already fails, or the severe end still passes, there is no boundary inside
    this range and saying so is the correct answer.

    ``verify_samples`` adds a uniform sweep afterwards to check the
    monotonicity that bisection assumed. It costs probes, so it is opt-in.
    """
    if not hi > lo:
        raise ValueError(f"hi ({hi}) must exceed lo ({lo})")
    if target_width <= 0.0:
        raise ValueError("target_width must be positive")

    probes: list[tuple[float, bool]] = []
    seen: dict[float, bool] = {}

    def probe(value: float) -> bool:
        """Evaluate once per distinct value; a repeat costs nothing."""
        for known, failed in seen.items():
            if math.isclose(value, known, rel_tol=1e-12, abs_tol=1e-12):
                return failed
        failed = evaluate(value)
        seen[value] = failed
        probes.append((value, failed))
        return failed

    if verify_samples > 1:
        # Coarse sweep first. Bisection alone only ever probes the endpoints
        # before concluding, so a failure band strictly inside the range -- both
        # endpoints passing -- would be reported as "no boundary" after two
        # probes. Scanning first is what makes that case detectable at all.
        step = (hi - lo) / (verify_samples - 1)
        for i in range(verify_samples):
            probe(lo + i * step)

        transitions = scan_boundaries(probes)
        if len(transitions) > 1:
            return BoundaryResult(
                status=BoundaryStatus.NOT_MONOTONE,
                probes_used=len(probes),
                probes=probes,
                transitions=transitions,
            )
        if not transitions:
            everywhere = probes[0][1]
            return BoundaryResult(
                status=(
                    BoundaryStatus.FAILS_THROUGHOUT
                    if everywhere
                    else BoundaryStatus.PASSES_THROUGHOUT
                ),
                probes_used=len(probes),
                probes=probes,
            )
        # Exactly one transition: refine inside the bracket the scan found.
        left, right = transitions[0]
    else:
        if probe(lo):
            return BoundaryResult(
                status=BoundaryStatus.FAILS_THROUGHOUT,
                probes_used=len(probes),
                probes=probes,
            )
        if not probe(hi):
            return BoundaryResult(
                status=BoundaryStatus.PASSES_THROUGHOUT,
                probes_used=len(probes),
                probes=probes,
            )
        left, right = lo, hi
    exhausted = False
    while right - left > target_width:
        if len(probes) >= max_probes:
            exhausted = True
            break
        middle = (left + right) / 2.0
        if probe(middle):
            right = middle
        else:
            left = middle

    if exhausted:
        return BoundaryResult(
            status=BoundaryStatus.BUDGET_EXHAUSTED,
            lower=left,
            upper=right,
            probes_used=len(probes),
            probes=probes,
        )

    return BoundaryResult(
        status=BoundaryStatus.LOCATED,
        lower=left,
        upper=right,
        probes_used=len(probes),
        probes=probes,
        transitions=scan_boundaries(probes),
    )
