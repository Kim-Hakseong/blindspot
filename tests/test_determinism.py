"""Rule T2 -- byte-identical replay.

If this file fails, every number the tool produces is unfalsifiable and no
other work proceeds. These tests are registry-driven: a newly registered
degradation is covered the moment it is registered, with no edit here.
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.degrade import REGISTRY

SEED = 4242

ALL = sorted(REGISTRY.names())


def _mid_params(deg):
    """Nominal parameters with every sweepable axis at its midpoint."""
    values = {}
    for axis in deg.axes:
        values[axis.field] = (axis.lo + axis.hi) / 2.0
    return deg.Params(**values)


def test_registry_is_not_empty():
    assert ALL, "no degradations registered"


@pytest.mark.parametrize("name", ALL)
def test_same_seed_is_byte_identical(name, scene):
    deg = REGISTRY.get(name)
    params = _mid_params(deg)
    a = deg.apply(scene, params, seed=SEED)
    b = deg.apply(scene, params, seed=SEED)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes(), f"{name} is not reproducible under a fixed seed"


@pytest.mark.parametrize("name", ALL)
def test_input_image_is_not_mutated(name, scene):
    """Rule T1: degradations are pure -- the caller's array is untouched."""
    deg = REGISTRY.get(name)
    before = scene.tobytes()
    deg.apply(scene, _mid_params(deg), seed=SEED)
    assert scene.tobytes() == before, f"{name} mutated its input image"


@pytest.mark.parametrize("name", ALL)
def test_global_rng_state_does_not_leak_in(name, scene):
    """Rule T1: no global RNG. Perturbing numpy's global state changes nothing."""
    deg = REGISTRY.get(name)
    params = _mid_params(deg)

    np.random.seed(1)
    a = deg.apply(scene, params, seed=SEED)
    np.random.seed(999)
    _ = np.random.random(1000)
    b = deg.apply(scene, params, seed=SEED)

    assert a.tobytes() == b.tobytes(), (
        f"{name} output depends on numpy's global RNG state; "
        "use np.random.default_rng(seed) only"
    )


@pytest.mark.parametrize("name", ALL)
def test_global_rng_state_is_not_disturbed(name, scene):
    """A degradation must not consume from the global RNG either."""
    deg = REGISTRY.get(name)
    np.random.seed(7)
    expected = np.random.random(5).copy()

    np.random.seed(7)
    deg.apply(scene, _mid_params(deg), seed=SEED)
    actual = np.random.random(5)

    assert np.array_equal(expected, actual), f"{name} consumed the global RNG"


@pytest.mark.parametrize("name", ALL)
def test_output_is_a_valid_image(name, scene):
    deg = REGISTRY.get(name)
    out = deg.apply(scene, _mid_params(deg), seed=SEED)
    assert out.dtype == np.uint8, f"{name} must return uint8"
    assert out.shape == scene.shape, f"{name} must preserve image shape"


@pytest.mark.parametrize("name", ALL)
def test_stochastic_kernels_vary_with_seed(name, scene):
    """Kernels that declare themselves stochastic must actually use the seed."""
    deg = REGISTRY.get(name)
    if not deg.stochastic:
        pytest.skip(f"{name} is deterministic by declaration")
    params = _mid_params(deg)
    a = deg.apply(scene, params, seed=1)
    b = deg.apply(scene, params, seed=2)
    assert a.tobytes() != b.tobytes(), (
        f"{name} declares stochastic=True but ignores the seed"
    )


@pytest.mark.parametrize("name", ALL)
def test_deterministic_kernels_ignore_seed(name, scene):
    deg = REGISTRY.get(name)
    if deg.stochastic:
        pytest.skip(f"{name} is stochastic by declaration")
    params = _mid_params(deg)
    a = deg.apply(scene, params, seed=1)
    b = deg.apply(scene, params, seed=98765)
    assert a.tobytes() == b.tobytes(), (
        f"{name} declares stochastic=False but its output depends on the seed"
    )
