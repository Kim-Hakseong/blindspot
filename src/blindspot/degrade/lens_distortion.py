"""Brown-Conrady lens distortion applied through ``cv::remap``.

The coefficients are the same k1, k2, p1, p2 OpenCV's own calibration produces,
so a user can take the distortion vector their camera calibration already
reported and ask what it costs them -- rather than inventing a warp strength.

Sign convention follows OpenCV: positive k1 is pincushion in the *undistort*
direction, so applying the model here produces the barrel/pincushion image an
uncalibrated pipeline would actually receive.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class LensDistortionParams(DegradeParams):
    k1: float = Field(default=0.0, ge=-1.0, le=1.0)
    k2: float = Field(default=0.0, ge=-1.0, le=1.0)
    p1: float = Field(default=0.0, ge=-0.1, le=0.1)
    p2: float = Field(default=0.0, ge=-0.1, le=0.1)


def _camera_matrix(width: int, height: int) -> np.ndarray:
    """A plausible intrinsic matrix: principal point centred, ~60 deg HFOV."""
    f = 0.9 * max(width, height)
    return np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        np.float64,
    )


@register
class LensDistortion(Degradation):
    name = "lens_distortion"
    Params = LensDistortionParams
    stochastic = False
    # Geometric: verified by a roundtrip test rather than a scalar image metric,
    # because distortion moves content without necessarily blurring it.
    axes = (
        Axis("k1", "dimensionless", 0.0, 0.4, expect=None),
        Axis("k2", "dimensionless", 0.0, 0.1, expect=None),
    )

    def _dist_coeffs(self, params: LensDistortionParams) -> np.ndarray:
        return np.array([params.k1, params.k2, params.p1, params.p2], np.float64)

    def derived(self, params: LensDistortionParams) -> dict[str, tuple[float, str]]:
        # Radial displacement at the frame corner for a unit-normalised radius,
        # expressed in px for a nominal 256 px frame so the number is legible.
        r = np.sqrt(2.0) / 2.0
        radial = params.k1 * r**2 + params.k2 * r**4
        return {"corner_displacement_px": (float(abs(radial) * 0.9 * 256.0 * r), "px")}

    def _maps(self, width: int, height: int, params: LensDistortionParams):
        k = _camera_matrix(width, height)
        return cv2.initUndistortRectifyMap(
            k, self._dist_coeffs(params), None, k, (width, height), cv2.CV_32FC1
        )

    def _apply(self, image, params: LensDistortionParams, rng) -> np.ndarray:
        h, w = image.shape[:2]
        map_x, map_y = self._maps(w, h, params)
        return cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    def roundtrip_error_px(
        self, width: int, height: int, params: LensDistortionParams
    ) -> float:
        """Max error, in px, of distorting then undistorting a grid of points.

        This is the honesty check on the warp: if the forward model and its
        inverse disagree, any boundary we report on this axis is measuring our
        own numerical error rather than the pipeline's tolerance.
        """
        k = _camera_matrix(width, height)
        d = self._dist_coeffs(params)

        ys, xs = np.mgrid[10 : height - 10 : 12, 10 : width - 10 : 12]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

        # Undistort back to normalised coords, then re-project through the model.
        undist = cv2.undistortPoints(pts.reshape(-1, 1, 2), k, d, P=None).reshape(-1, 2)
        obj = np.concatenate([undist, np.ones((len(undist), 1))], axis=1)
        reproj, _ = cv2.projectPoints(
            obj, np.zeros(3), np.zeros(3), k, d
        )
        return float(np.abs(reproj.reshape(-1, 2) - pts).max())
