"""Atmospheric scattering (fog/haze) via the Koschmieder model.

The standard image formation model for a scattering medium is

    I(x) = J(x) * t(x) + A * (1 - t(x))

where J is the clear scene radiance, A is the atmospheric light, and the
transmission follows Beer-Lambert over the depth d(x) of each pixel:

    t(x) = exp(-beta * d(x))

beta is the extinction coefficient in 1/m -- the quantity meteorological
visibility is defined from (the Koschmieder relation puts visibility at
roughly 3.912 / beta for a 2% contrast threshold). Reporting a fog boundary in
beta therefore converts directly to "this pipeline fails below N metres of
visibility", which is a statement an operator can act on.

Depth is approximated as a linear ramp increasing towards the top of the frame,
the standard assumption for a forward-facing camera on a ground plane. That
approximation is a real limitation and is recorded as one.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from .base import Axis, DegradeParams, Degradation, register


class FogParams(DegradeParams):
    beta_per_m: float = Field(ge=0.0, le=3.0)
    depth_near_m: float = Field(default=5.0, gt=0.0, le=1000.0)
    depth_far_m: float = Field(default=80.0, gt=0.0, le=5000.0)
    airlight: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _depth_ordered(self):
        if self.depth_far_m <= self.depth_near_m:
            raise ValueError("depth_far_m must exceed depth_near_m")
        return self


@register
class Fog(Degradation):
    name = "fog"
    Params = FogParams
    stochastic = False
    axes = (Axis("beta_per_m", "1/m", 0.0, 0.12, expect=("rms_contrast", "decreasing")),)

    def derived(self, params: FogParams) -> dict[str, tuple[float, str]]:
        t_near = float(np.exp(-params.beta_per_m * params.depth_near_m))
        t_far = float(np.exp(-params.beta_per_m * params.depth_far_m))
        # Koschmieder visibility at the 2% contrast threshold.
        visibility = float(3.912 / params.beta_per_m) if params.beta_per_m > 1e-9 else float("inf")
        out = {
            "transmission_near": (t_near, "dimensionless"),
            "transmission_far": (t_far, "dimensionless"),
        }
        if np.isfinite(visibility):
            out["meteorological_visibility_m"] = (visibility, "m")
        return out

    def _apply(self, image, params: FogParams, rng) -> np.ndarray:
        if params.beta_per_m <= 0.0:
            return image.copy()

        h, w = image.shape[:2]
        # Depth increases towards the top of the frame (ground-plane camera).
        depth = np.linspace(params.depth_far_m, params.depth_near_m, h, dtype=np.float32)
        t = np.exp(-params.beta_per_m * depth)[:, None, None]

        j = image.astype(np.float32) / 255.0
        i = j * t + params.airlight * (1.0 - t)
        return np.clip(i * 255.0, 0, 255).astype(np.uint8)
