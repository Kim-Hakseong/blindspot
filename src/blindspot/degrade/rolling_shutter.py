"""Rolling-shutter skew from sensor readout time and camera angular velocity.

A CMOS sensor with an electronic rolling shutter exposes rows sequentially. If
the camera rotates during readout, row r is captured at time

    t(r) = (r / H) * T_readout

and so is displaced relative to row 0 by

    dx(r) = f_px * tan(omega * t(r))

The result is the shear that makes vertical poles lean in footage from a moving
vehicle. This is a per-row ``cv::remap``, not a global affine shear, because
the displacement is a function of row time and stays exact if the relationship
is made non-linear later (e.g. adding readout blanking).
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class RollingShutterParams(DegradeParams):
    readout_time_ms: float = Field(ge=0.0, le=100.0)
    angular_velocity_deg_s: float = Field(default=60.0, ge=-720.0, le=720.0)
    focal_length_px: float = Field(default=900.0, gt=0.0, le=20000.0)
    height_px: float = Field(default=256.0, gt=0.0, le=8192.0)


@register
class RollingShutter(Degradation):
    name = "rolling_shutter"
    Params = RollingShutterParams
    stochastic = False
    # Geometric: content moves rather than degrades, so this is verified by the
    # dedicated shear test in tests/test_degrade.py.
    axes = (
        Axis("readout_time_ms", "ms", 0.0, 40.0, expect=None),
        Axis("angular_velocity_deg_s", "deg/s", 0.0, 240.0, expect=None),
    )

    def derived(self, params: RollingShutterParams) -> dict[str, tuple[float, str]]:
        theta = np.deg2rad(params.angular_velocity_deg_s * params.readout_time_ms / 1000.0)
        total = float(params.focal_length_px * np.tan(theta))
        return {
            "row_shift_total_px": (total, "px"),
            "row_shift_per_row_px": (total / float(params.height_px), "px"),
        }

    def _apply(self, image, params: RollingShutterParams, rng) -> np.ndarray:
        h, w = image.shape[:2]
        total = self.derived(params)["row_shift_total_px"][0]
        if abs(total) < 1e-9:
            return image.copy()

        # Displacement grows linearly with row readout time.
        rows = np.arange(h, dtype=np.float32)
        shift = (rows / float(h)) * np.float32(total)

        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + shift[:, None]
        map_y = np.tile(rows[:, None], (1, w))
        return cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
