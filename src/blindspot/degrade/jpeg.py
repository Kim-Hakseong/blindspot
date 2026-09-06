"""JPEG recompression at a given libjpeg quality.

There is no modelling to do here and that is the point: the axis value *is* the
quality setting a CDN, a camera, or an upstream service would use, so a
boundary at q=38 is directly actionable. This calls OpenCV's own encoder so the
quantisation tables are the standard ones rather than an approximation of them.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class JpegParams(DegradeParams):
    quality: float = Field(ge=1.0, le=100.0)


@register
class Jpeg(Degradation):
    name = "jpeg"
    Params = JpegParams
    stochastic = False
    axes = (Axis("quality", "jpeg-q", 5.0, 100.0, expect=("blockiness", "decreasing")),)

    def derived(self, params: JpegParams) -> dict[str, tuple[float, str]]:
        return {}

    def _apply(self, image, params: JpegParams, rng) -> np.ndarray:
        quality = int(round(params.quality))
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG decode failed")
        return decoded
