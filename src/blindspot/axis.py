"""A sweepable physical axis.

Kept free of OpenCV and every kernel so that code which only needs to reason
about axes -- the round planner in its Lambda, for one -- can import it without
pulling in the image stack. `blindspot.degrade` re-exports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What an objective measurement should do as an axis value increases.
Direction = Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class Axis:
    """A single sweepable physical dimension of a degradation.

    ``field``      -- the parameter this axis moves
    ``unit``       -- the physical unit it is expressed in
    ``lo/hi``      -- the range the boundary search is allowed to explore
    ``expect``     -- ``(metric, direction)`` the sweep must objectively
                      produce, or ``None`` for geometric kernels verified by a
                      dedicated test instead of by a scalar image measurement.
    ``severe_end`` -- which end of the range is the *harsh* condition. Not all
                      axes get worse as the number goes up: more exposure and
                      more fog are worse, but more light and higher JPEG
                      quality are better. A search that assumes severity always
                      increases with the value will report "no boundary" on
                      half the axes, because it walks from harsh to benign and
                      never sees a pass-then-fail transition.
    """

    field: str
    unit: str
    lo: float
    hi: float
    expect: tuple[str, Direction] | None = None
    severe_end: Literal["lo", "hi"] = "hi"

    def clamp(self, value: float) -> float:
        return float(min(max(value, self.lo), self.hi))

    @property
    def benign(self) -> float:
        """The mild end of the range."""
        return self.lo if self.severe_end == "hi" else self.hi

    @property
    def severe(self) -> float:
        """The harsh end of the range."""
        return self.hi if self.severe_end == "hi" else self.lo

    def to_severity(self, value: float) -> float:
        """Map an axis value onto [0, 1], 0 benign and 1 severe."""
        return (value - self.benign) / (self.severe - self.benign)

    def from_severity(self, severity: float) -> float:
        """Map a [0, 1] severity back onto the axis's physical unit."""
        return self.benign + severity * (self.severe - self.benign)
