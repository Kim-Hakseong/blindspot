"""Coverage inference.

The sweep that native measurements are inverted through must describe the
population, not one frame: sharpness and contrast are scene-dependent, so a
single-frame curve reads another scene's texture as a different condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from blindspot.boundary.coverage import axis_coverage, population_sweep


@dataclass
class F:
    image: np.ndarray
    image_id: str = "x"


def frames(scene, n=4):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        shifted = np.roll(scene, int(rng.integers(0, 64)), axis=1)
        out.append(F(shifted, str(i)))
    return out


def test_population_sweep_is_monotone_on_a_monotone_axis(scene):
    values = [0.0, 10.0, 20.0, 40.0]
    medians = population_sweep(frames(scene), "motion_blur", "exposure_ms",
                               values, "laplacian_var", seed=1)
    assert medians == sorted(medians, reverse=True)


def test_population_sweep_is_deterministic(scene):
    args = (frames(scene), "low_light", "illuminance_lux", [1.0, 50.0, 400.0], "snr_db")
    assert population_sweep(*args, seed=3) == population_sweep(*args, seed=3)


def test_population_sweep_uses_the_median_not_the_first_frame(scene):
    fs = frames(scene)
    fs[0] = F(np.full_like(scene, 128), "flat")  # a textureless outlier first
    medians = population_sweep(fs, "motion_blur", "exposure_ms", [0.0], "laplacian_var", seed=1)
    assert medians[0] > 1.0, "a flat first frame must not define the curve"


def test_full_coverage_when_native_spans_the_sweep():
    cov = axis_coverage("a.b", "ms", (0.0, 40.0), "m", [0, 20, 40], [100, 50, 10], [100, 10])
    assert cov.coverage_fraction == pytest.approx(1.0)
    assert cov.uncovered_regions == []


def test_pristine_set_covers_only_the_benign_end():
    cov = axis_coverage("a.b", "ms", (0.0, 40.0), "m", [0, 20, 40], [100, 50, 10], [100, 95])
    assert cov.coverage_fraction < 0.1
    assert cov.uncovered_regions[-1]["to"] == pytest.approx(40.0)
