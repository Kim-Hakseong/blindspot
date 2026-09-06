"""Motion blur derived from exposure time and camera angular velocity.

The point of this kernel is that its parameters are things you can look up or
measure, not a blur radius someone chose because the picture looked right.

A camera rotating at omega (deg/s) during an exposure of t (ms) sweeps an
angle theta = omega * t. A feature at the optical axis is displaced across the
sensor by

    s = f_px * tan(theta)

where f_px is the focal length expressed in pixels. That displacement is the
support of the point spread function: the scene point is smeared along a line
s pixels long. We build exactly that PSF -- a normalised line -- and convolve.

A box PSF is the correct first-order model for constant angular velocity over
the exposure, which is also why the ends are weighted for the fractional part
of the length rather than rounded: a 9.43 px smear is not a 9 px smear.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class MotionBlurParams(DegradeParams):
    exposure_ms: float = Field(ge=0.0, le=1000.0)
    angular_velocity_deg_s: float = Field(ge=0.0, le=720.0)
    # Capture context. Defaults describe a 1/2.8" sensor with a 4 mm lens at
    # 1280x720, i.e. a common fixed edge camera; override to match your rig.
    focal_length_px: float = Field(default=900.0, gt=0.0, le=20000.0)
    direction_deg: float = Field(default=0.0, ge=-180.0, le=360.0)


def _line_psf(length_px: float, direction_deg: float) -> np.ndarray:
    """A normalised line PSF of the given sub-pixel length and orientation.

    The line is rasterised by accumulating unit mass along it at sub-pixel
    steps, so a length of 9.43 px carries strictly more smear than 9.0 px.
    """
    size = int(np.ceil(length_px)) + 3
    size += 1 - size % 2  # odd, so the kernel has a true centre
    psf = np.zeros((size, size), np.float64)
    centre = (size - 1) / 2.0

    theta = np.deg2rad(direction_deg)
    dx, dy = np.cos(theta), -np.sin(theta)

    # Sample the segment densely and splat with bilinear weights.
    n = max(int(np.ceil(length_px * 16.0)), 1)
    for t in np.linspace(-length_px / 2.0, length_px / 2.0, n):
        x, y = centre + dx * t, centre + dy * t
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if 0 <= yy < size and 0 <= xx < size:
                    psf[yy, xx] += wy * wx

    total = psf.sum()
    if total <= 0.0:
        psf[int(centre), int(centre)] = 1.0
        return psf
    return psf / total


@register
class MotionBlur(Degradation):
    name = "motion_blur"
    Params = MotionBlurParams
    stochastic = False
    axes = (
        Axis("exposure_ms", "ms", 0.0, 40.0, expect=("laplacian_var", "decreasing")),
        Axis(
            "angular_velocity_deg_s",
            "deg/s",
            0.0,
            120.0,
            expect=("laplacian_var", "decreasing"),
        ),
    )

    def derived(self, params: MotionBlurParams) -> dict[str, tuple[float, str]]:
        theta_rad = np.deg2rad(params.angular_velocity_deg_s * params.exposure_ms / 1000.0)
        psf_px = float(params.focal_length_px * np.tan(theta_rad))
        return {
            "psf_length_px": (psf_px, "px"),
            "sweep_angle_deg": (
                float(params.angular_velocity_deg_s * params.exposure_ms / 1000.0),
                "deg",
            ),
        }

    def _apply(self, image, params: MotionBlurParams, rng) -> np.ndarray:
        length = self.derived(params)["psf_length_px"][0]
        if length < 1e-6:
            return image.copy()
        psf = _line_psf(length, params.direction_deg).astype(np.float32)
        return cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT_101)
