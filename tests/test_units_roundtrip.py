"""Rule T3 -- degradation strength is expressed in physical units, and moving
a physical parameter moves an objective measurement in the predicted direction.

This is the test that keeps the report comparable to a camera spec sheet. A
kernel whose axis does not measurably do what its unit claims is a kernel we
cannot put a number on.
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.degrade import REGISTRY
from blindspot.measure import measure

SEED = 31337

# Units a capture engineer can look up. Anything outside this set is a smell:
# "intensity", "strength", "level", "amount" would all be rejected here.
ALLOWED_UNITS = {
    "ms",
    "deg/s",
    "deg",
    "px",
    "lux",
    "m",
    "mm",
    "um",
    "1/m",
    "f-number",
    "jpeg-q",
    "crf",
    "cy/px",
    "dimensionless",
}

BANNED_UNIT_WORDS = ("intensity", "strength", "severity", "level", "amount", "scale")

AXES = [(name, axis) for name in sorted(REGISTRY.names()) for axis in REGISTRY.get(name).axes]
AXIS_IDS = [f"{name}.{axis.field}" for name, axis in AXES]


def _params_with(deg, axis, value):
    values = {a.field: (a.lo + a.hi) / 2.0 for a in deg.axes}
    values[axis.field] = value
    return deg.Params(**values)


@pytest.mark.parametrize("name,axis", AXES, ids=AXIS_IDS)
def test_axis_declares_a_physical_unit(name, axis):
    assert axis.unit in ALLOWED_UNITS, (
        f"{name}.{axis.field} uses unit {axis.unit!r}, which is not a physical unit"
    )
    lowered = f"{axis.field} {axis.unit}".lower()
    for banned in BANNED_UNIT_WORDS:
        assert banned not in lowered, (
            f"{name}.{axis.field} looks like an arbitrary scale ({banned!r}); "
            "rule T3 requires physical units"
        )


@pytest.mark.parametrize("name,axis", AXES, ids=AXIS_IDS)
def test_axis_range_is_ordered(name, axis):
    assert axis.lo < axis.hi


@pytest.mark.parametrize("name,axis", AXES, ids=AXIS_IDS)
def test_axis_moves_its_measurement_monotonically(name, axis, scene):
    """Sweep the physical axis; the declared measurement must move as declared.

    Monotonicity is checked on the endpoints plus a rank correlation across the
    sweep, so a kernel that is merely noisy in the middle still passes while a
    kernel that does nothing, or moves the wrong way, fails.
    """
    if axis.expect is None:
        pytest.skip(f"{name}.{axis.field} is verified by a dedicated geometric test")

    metric_name, direction = axis.expect
    deg = REGISTRY.get(name)

    values = np.linspace(axis.lo, axis.hi, 7)
    measured = [
        measure(deg.apply(scene, _params_with(deg, axis, v), seed=SEED))[metric_name]
        for v in values
    ]

    sign = 1.0 if direction == "increasing" else -1.0
    first, last = measured[0], measured[-1]

    assert sign * (last - first) > 0, (
        f"{name}.{axis.field} ({axis.unit}) should drive {metric_name} "
        f"{direction} across [{axis.lo}, {axis.hi}], but it went "
        f"{first:.4g} -> {last:.4g}"
    )

    # Rank correlation: tolerate local wobble, reject a non-trend.
    ranks = np.argsort(np.argsort(np.array(measured) * sign))
    corr = np.corrcoef(ranks, np.arange(len(values)))[0, 1]
    assert corr > 0.8, (
        f"{name}.{axis.field} does not move {metric_name} monotonically "
        f"(rank corr {corr:.2f}); measured={['%.4g' % m for m in measured]}"
    )


@pytest.mark.parametrize("name,axis", AXES, ids=AXIS_IDS)
def test_axis_low_end_is_close_to_a_no_op(name, axis, scene):
    """The bottom of a physical range must be a mild condition, not a broken one.

    Guards against a kernel whose 'lo' already destroys the image, which would
    make any boundary found on that axis meaningless.
    """
    if axis.expect is None:
        pytest.skip(f"{name}.{axis.field} is verified by a dedicated geometric test")

    deg = REGISTRY.get(name)
    benign = axis.lo if axis.expect[1] == "decreasing" else axis.hi
    out = deg.apply(scene, _params_with(deg, axis, benign), seed=SEED)

    diff = np.abs(out.astype(np.int16) - scene.astype(np.int16)).mean()
    assert diff < 40.0, (
        f"{name}.{axis.field} at its benign end ({benign} {axis.unit}) already "
        f"changes the image by {diff:.1f} grey levels on average"
    )


def test_derived_quantities_are_reported_with_units():
    """Kernels that derive a quantity (e.g. PSF length in px) must label it."""
    for name in sorted(REGISTRY.names()):
        deg = REGISTRY.get(name)
        params = deg.Params(**{a.field: (a.lo + a.hi) / 2.0 for a in deg.axes})
        for key, (value, unit) in deg.derived(params).items():
            assert unit in ALLOWED_UNITS, f"{name}.{key} has non-physical unit {unit!r}"
            assert np.isfinite(value), f"{name}.{key} is not finite"
