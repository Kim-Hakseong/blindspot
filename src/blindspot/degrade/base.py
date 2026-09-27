"""The degradation contract.

Every degradation is a pure function of ``(image, params, seed)``. There is no
module-level random state anywhere in this package: stochastic kernels take a
generator built from the caller's seed and nothing else. That is what makes a
failure in the report regenerable byte-for-byte, which is in turn what makes
the report a measurement rather than an anecdote.

Parameters are physical. A kernel exposes exposure time in milliseconds,
angular velocity in degrees per second, illuminance in lux, an extinction
coefficient in 1/m -- quantities a capture engineer can read off a spec sheet
and compare against. There is deliberately no "intensity 0..1" anywhere.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..axis import Axis, Direction  # re-exported: the axis contract has no OpenCV dependency

class DegradeParams(BaseModel):
    """Base for kernel parameter sets.

    Frozen so a parameter vector cannot drift after it has been recorded in the
    ledger, and ``extra="forbid"`` so an arbitrary knob cannot be smuggled in
    alongside the physical ones.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Degradation(abc.ABC):
    """A physically parameterised capture-condition degradation."""

    #: Registry key, e.g. ``"motion_blur"``.
    name: ClassVar[str]
    #: Parameter model for this kernel.
    Params: ClassVar[type[DegradeParams]]
    #: Sweepable physical axes.
    axes: ClassVar[tuple[Axis, ...]]
    #: Whether the kernel consumes randomness. Enforced both ways by tests.
    stochastic: ClassVar[bool] = False

    def apply(self, image: np.ndarray, params: DegradeParams, seed: int) -> np.ndarray:
        """Return a degraded copy of ``image``. Never mutates the input."""
        if image.dtype != np.uint8:
            raise TypeError(f"{self.name}: expected uint8 image, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{self.name}: expected an HxWx3 BGR image, got {image.shape}")

        rng = np.random.default_rng(seed)
        out = self._apply(image, params, rng)

        if out.dtype != np.uint8 or out.shape != image.shape:
            raise AssertionError(f"{self.name}: kernel broke the image contract")
        return out

    @abc.abstractmethod
    def _apply(
        self, image: np.ndarray, params: DegradeParams, rng: np.random.Generator
    ) -> np.ndarray:
        """Kernel implementation. Must not touch global random state."""

    def derived(self, params: DegradeParams) -> dict[str, tuple[float, str]]:
        """Quantities computed from the parameters, as ``{name: (value, unit)}``.

        This is how a kernel reports the thing the user actually cares about --
        a PSF length in pixels, a circle of confusion in pixels -- alongside
        the capture settings that produced it.
        """
        return {}

    def axis(self, field: str) -> Axis:
        for a in self.axes:
            if a.field == field:
                return a
        raise KeyError(f"{self.name} has no axis {field!r}")

    def nominal(self) -> DegradeParams:
        """Every axis at the midpoint of its range."""
        return self.Params(**{a.field: (a.lo + a.hi) / 2.0 for a in self.axes})


class _Registry:
    """Name -> degradation instance. Populated by ``@register`` at import."""

    def __init__(self) -> None:
        self._items: dict[str, Degradation] = {}

    def add(self, deg: Degradation) -> None:
        if deg.name in self._items:
            raise ValueError(f"duplicate degradation name: {deg.name}")
        self._items[deg.name] = deg

    def get(self, name: str) -> Degradation:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown degradation {name!r}; registered: {sorted(self._items)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


REGISTRY = _Registry()


def register(cls: type[Degradation]) -> type[Degradation]:
    """Class decorator: make a degradation discoverable by name."""
    REGISTRY.add(cls())
    return cls
