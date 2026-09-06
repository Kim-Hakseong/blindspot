"""Physically parameterised capture-condition degradations, built on OpenCV 5.

Importing this package registers every kernel, so ``REGISTRY`` is the single
source of truth for which axes exist. Tests are driven off it, which means a
newly registered kernel is held to the determinism and physical-unit contracts
automatically rather than by remembering to add a test.
"""

from __future__ import annotations

from .base import REGISTRY, Axis, DegradeParams, Degradation, register

# Import for side effect: each module registers its kernel.
from . import defocus, fog, jpeg, lens_distortion, low_light, motion_blur, rolling_shutter  # noqa: E402,F401

__all__ = ["REGISTRY", "Axis", "DegradeParams", "Degradation", "register"]
