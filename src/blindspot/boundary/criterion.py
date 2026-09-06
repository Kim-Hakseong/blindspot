"""What counts as failure.

The definition lives here and nowhere else. A tool whose pass/fail threshold is
duplicated across modules will eventually report a boundary under one
definition and a curve under another, and the discrepancy will be invisible.

Failure is *relative to the pipeline's own undegraded baseline*, not an
absolute score. A pipeline that starts at mAP 0.61 and one that starts at 0.85
should both be described by where they lose a given fraction of their own
capability; an absolute threshold would call the first one broken before any
degradation was applied.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A condition fails when mAP@50 falls below this fraction of the undegraded
#: baseline. 0.60 means "lost 40% of its own capability".
FAILURE_FRACTION = 0.60


@dataclass(frozen=True)
class Criterion:
    """The pass/fail rule for one run, fixed when the run starts."""

    baseline_map50: float
    fraction: float = FAILURE_FRACTION

    @property
    def threshold(self) -> float:
        """The absolute mAP@50 below which a condition is a failure."""
        return self.baseline_map50 * self.fraction

    def failed(self, map50: float) -> bool:
        return map50 < self.threshold

    def describe(self) -> dict[str, float | str]:
        return {
            "rule": "mAP@50 below a fraction of the undegraded baseline",
            "baseline_map50": self.baseline_map50,
            "fraction": self.fraction,
            "threshold_map50": self.threshold,
        }
