"""Shared fixtures.

Every fixture here is deterministic. A test image is generated from a fixed
seed rather than read from disk so the test suite has no data dependency and
so degradation measurements have known, broadband content to act on.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def _synthetic_scene(size: int = 256, seed: int = 20260906) -> np.ndarray:
    """A BGR scene with content at many spatial frequencies.

    Contains: a luminance ramp (low frequency), nested checkerboards (mid to
    high frequency), diagonal bars (orientation content), and a few filled
    rectangles with hard edges. Deliberately *not* random noise, because
    sharpness measures on white noise are dominated by the noise floor.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)

    # Low-frequency luminance ramp.
    ramp = np.linspace(40, 200, size, dtype=np.float32)
    img += ramp[None, :, None]

    # Nested checkerboards at three scales -> mid/high frequency energy.
    # Cell sizes are deliberately coprime with 8 so that scene content does not
    # sit on the JPEG block grid and inflate the blockiness measurement.
    yy, xx = np.mgrid[0:size, 0:size]
    for cell, amp in ((35, 40.0), (13, 25.0), (5, 15.0)):
        checker = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
        img += ((checker - 0.5) * 2.0 * amp)[..., None]

    # Hard-edged rectangles: sharp step edges for blur to act on.
    for _ in range(6):
        x0, y0 = rng.integers(8, size - 72, size=2)
        w, h = rng.integers(24, 64, size=2)
        colour = rng.uniform(30, 220, size=3).astype(np.float32)
        img[y0 : y0 + h, x0 : x0 + w] = colour

    # Slight per-channel offset so colour channels are not identical.
    img += np.array([6.0, 0.0, -6.0], np.float32)[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    """A 256x256 BGR uint8 test scene."""
    return _synthetic_scene()


@pytest.fixture(scope="session")
def slanted_edge() -> np.ndarray:
    """A 5-degree slanted dark/light edge -- the ISO 12233 style MTF target."""
    size = 256
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = np.deg2rad(5.0)
    # Signed distance from a line through the centre, tilted by `angle`.
    d = (xx - size / 2.0) * np.cos(angle) - (yy - size / 2.0) * np.sin(angle)
    edge = np.where(d > 0, 220.0, 35.0).astype(np.float32)
    return cv2.cvtColor(edge.astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="session")
def grid() -> np.ndarray:
    """A regular grid of lines -- used to verify geometric warps."""
    size = 256
    img = np.full((size, size, 3), 20, np.uint8)
    for i in range(0, size, 16):
        img[i : i + 2, :] = 235
        img[:, i : i + 2] = 235
    return img
