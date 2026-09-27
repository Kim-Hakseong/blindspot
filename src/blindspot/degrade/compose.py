"""A capture condition made of several degradations at once.

Real footage is rarely wrong in one way. Dusk brings low light *and* the long
exposure that causes motion blur. A composite is applied in the order light
actually travels:

    atmosphere  ->  optics / motion  ->  sensor  ->  codec
    (fog)          (blur, defocus,     (low light)   (JPEG)
                    distortion,
                    rolling shutter)

so the caller's argument order cannot change the result. Each step gets its
own seed, derived from the run seed and the step's position, so two stochastic
steps never share a noise stream and the composite stays a pure function of
``(image, condition, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import REGISTRY

#: Position of each kernel in the capture chain.
STAGE = {
    "fog": 0,
    "motion_blur": 1,
    "defocus": 1,
    "lens_distortion": 1,
    "rolling_shutter": 1,
    "low_light": 2,
    "jpeg": 3,
}


#: Parameter-name suffix -> physical unit. Parameters are named for their unit
#: throughout the package, so this is a naming contract rather than a lookup.
_SUFFIX_UNITS = (
    ("_deg_s", "deg/s"),
    ("_per_m", "1/m"),
    ("_ms", "ms"),
    ("_lux", "lux"),
    ("_px", "px"),
    ("_mm", "mm"),
    ("_um", "um"),
    ("_deg", "deg"),
    ("_m", "m"),
)


def unit_of(field: str) -> str:
    for suffix, unit in _SUFFIX_UNITS:
        if field.endswith(suffix):
            return unit
    return "dimensionless"


@dataclass(frozen=True)
class Step:
    degradation: str
    values: tuple[tuple[str, float], ...]
    #: Non-axis parameters pinned for this condition, e.g. a shared exposure.
    fixed: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Condition:
    steps: tuple[Step, ...]

    @classmethod
    def from_axes(
        cls, axes: dict[str, float], fixed: dict[str, float] | None = None
    ) -> "Condition":
        """Build from ``{"motion_blur.exposure_ms": 12.0, ...}``.

        ``fixed`` pins non-axis parameters, which is how two kernels are tied
        to one physical quantity: a camera has a single exposure time, and it
        sets both the blur length and the light the sensor collects.
        """
        grouped: dict[str, dict[str, float]] = {}
        for axis_id, value in axes.items():
            degradation, _, field = axis_id.partition(".")
            REGISTRY.get(degradation).axis(field)  # raises KeyError if unknown
            grouped.setdefault(degradation, {})[field] = float(value)
        pinned: dict[str, dict[str, float]] = {}
        for param_id, value in (fixed or {}).items():
            degradation, _, field = param_id.partition(".")
            if field not in REGISTRY.get(degradation).Params.model_fields:
                raise KeyError(f"{degradation} has no parameter {field!r}")
            pinned.setdefault(degradation, {})[field] = float(value)
            grouped.setdefault(degradation, {})
        ordered = sorted(grouped, key=lambda name: (STAGE[name], name))
        return cls(tuple(
            Step(n, tuple(sorted(grouped[n].items())), tuple(sorted(pinned.get(n, {}).items())))
            for n in ordered
        ))

    def params(self, step: Step):
        deg = REGISTRY.get(step.degradation)
        values = {a.field: (a.lo + a.hi) / 2.0 for a in deg.axes}
        values.update(dict(step.fixed))
        values.update(dict(step.values))
        return deg.Params(**values)

    def describe(self) -> list[dict]:
        out = []
        for step in self.steps:
            deg = REGISTRY.get(step.degradation)
            for field, value in step.values:
                out.append({"axis": f"{step.degradation}.{field}", "value": value,
                            "unit": deg.axis(field).unit})
            for field, value in step.fixed:
                out.append({"axis": f"{step.degradation}.{field}", "value": value,
                            "unit": unit_of(field), "fixed": True})
        return out


def _step_seed(seed: int, index: int) -> int:
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def apply_condition(image: np.ndarray, condition: Condition, seed: int) -> np.ndarray:
    """Apply every step in physical order. Never mutates ``image``."""
    if len(condition.steps) == 1:
        # A single step uses the run seed unchanged, so a 1-D probe and the
        # same condition expressed as a composite are the same computation.
        step = condition.steps[0]
        return REGISTRY.get(step.degradation).apply(image, condition.params(step), seed=seed)

    out = image
    for index, step in enumerate(condition.steps):
        deg = REGISTRY.get(step.degradation)
        out = deg.apply(out, condition.params(step), seed=_step_seed(seed, index))
    return out
