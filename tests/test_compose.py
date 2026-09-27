"""Composing degradations into one capture condition.

Two degradations applied together are only meaningful in the order the
physics imposes -- the atmosphere acts before the lens, the lens before the
sensor, the sensor before the codec -- and the composite must obey the same
determinism contract as a single kernel (rule T2).
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.degrade import REGISTRY
from blindspot.degrade.compose import STAGE, Condition, apply_condition


def cond(**axes):
    return Condition.from_axes(axes)


def test_every_registered_kernel_has_a_physical_stage():
    for name in REGISTRY.names():
        assert name in STAGE, f"{name} has no stage in the capture chain"


def test_composite_is_byte_identical_under_a_fixed_seed(scene):
    c = cond(**{"motion_blur.exposure_ms": 12.0, "low_light.illuminance_lux": 20.0})
    a = apply_condition(scene, c, seed=7)
    b = apply_condition(scene, c, seed=7)
    assert a.tobytes() == b.tobytes()


def test_composite_does_not_mutate_input(scene):
    before = scene.tobytes()
    apply_condition(scene, cond(**{"fog.beta_per_m": 0.05, "jpeg.quality": 20.0}), seed=1)
    assert scene.tobytes() == before


def test_order_follows_physics_not_argument_order(scene):
    """Blur-then-noise and noise-then-blur differ. Whatever order the caller
    lists them in, the sensor must act after the optics."""
    ab = Condition.from_axes({"motion_blur.exposure_ms": 20.0, "low_light.illuminance_lux": 5.0})
    ba = Condition.from_axes({"low_light.illuminance_lux": 5.0, "motion_blur.exposure_ms": 20.0})
    assert apply_condition(scene, ab, seed=3).tobytes() == apply_condition(scene, ba, seed=3).tobytes()
    assert [s.degradation for s in ab.steps] == ["motion_blur", "low_light"]


def test_single_axis_condition_equals_the_kernel_alone(scene):
    """Composition must not perturb a single kernel -- otherwise 1-D and 2-D
    results would describe different conditions."""
    deg = REGISTRY.get("motion_blur")
    params = deg.Params(**{a.field: (a.lo + a.hi) / 2.0 for a in deg.axes} | {"exposure_ms": 9.0})
    direct = deg.apply(scene, params, seed=5)
    composed = apply_condition(scene, cond(**{"motion_blur.exposure_ms": 9.0}), seed=5)
    assert direct.tobytes() == composed.tobytes()


def test_stochastic_steps_get_independent_streams(scene):
    """Two stochastic steps must not share a noise stream; and changing the
    run seed must change the result."""
    c = cond(**{"low_light.illuminance_lux": 5.0})
    assert apply_condition(scene, c, seed=1).tobytes() != apply_condition(scene, c, seed=2).tobytes()


def test_unknown_axis_is_rejected():
    with pytest.raises(KeyError):
        cond(**{"motion_blur.intensity": 0.5})


def test_condition_describes_itself_in_physical_units():
    c = cond(**{"motion_blur.exposure_ms": 12.0, "low_light.illuminance_lux": 20.0})
    described = c.describe()
    assert described == [
        {"axis": "motion_blur.exposure_ms", "value": 12.0, "unit": "ms"},
        {"axis": "low_light.illuminance_lux", "value": 20.0, "unit": "lux"},
    ]


def test_fixed_parameters_couple_kernels_to_one_physical_quantity(scene):
    """One exposure time drives both blur and sensor signal in a real camera."""
    c = Condition.from_axes(
        {"motion_blur.exposure_ms": 30.0, "low_light.illuminance_lux": 50.0},
        fixed={"low_light.exposure_ms": 30.0},
    )
    step = next(s for s in c.steps if s.degradation == "low_light")
    assert c.params(step).exposure_ms == 30.0
    assert {"axis": "low_light.exposure_ms", "value": 30.0, "unit": "ms", "fixed": True} in c.describe()


def test_fixed_parameter_must_exist_on_the_kernel():
    with pytest.raises(KeyError):
        Condition.from_axes({"low_light.illuminance_lux": 5.0}, fixed={"low_light.nonsense": 1.0})


def test_unit_naming_contract_holds_for_every_axis():
    """Axis units must agree with the parameter-name suffix convention."""
    from blindspot.degrade.compose import unit_of

    special = {"jpeg-q", "f-number", "dimensionless"}
    for name in REGISTRY.names():
        for axis in REGISTRY.get(name).axes:
            if axis.unit in special:
                continue
            assert unit_of(axis.field) == axis.unit, f"{name}.{axis.field}"
