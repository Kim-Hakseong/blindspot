"""Defocus blur derived from the thin-lens circle of confusion.

For a lens of focal length f at aperture N, focused at distance S1, a subject
at distance S2 images as a disc rather than a point. The diameter of that disc
-- the circle of confusion -- is

    CoC = |S2 - S1| / S2  *  f^2 / (N * (S1 - f))

with all distances in the same units. Divided by the sensor pixel pitch this
gives a diameter in pixels, which is the actual support of the blur.

The PSF of a defocused circular aperture is a uniform disc, not a Gaussian.
That distinction matters here: a disc PSF has zeros in its transfer function,
so it destroys certain spatial frequencies outright and produces the contrast
reversals real out-of-focus optics produce. Substituting a Gaussian would make
the MTF curve wrong in exactly the region a lens spec sheet describes.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class DefocusParams(DegradeParams):
    subject_distance_m: float = Field(gt=0.0, le=1000.0)
    aperture_f_number: float = Field(default=2.8, gt=0.0, le=32.0)
    # Optical context. Defaults describe an 8 mm lens on a 2 um-pitch sensor
    # focused at 5 m -- a typical fixed industrial/edge camera.
    focal_length_mm: float = Field(default=8.0, gt=0.0, le=600.0)
    focus_distance_m: float = Field(default=20.0, gt=0.0, le=1000.0)
    pixel_pitch_um: float = Field(default=2.0, gt=0.0, le=20.0)


def _disc_psf(diameter_px: float) -> np.ndarray:
    """A normalised uniform disc of the given diameter, anti-aliased.

    Supersampled 4x so that a 2.4 px disc differs from a 2.0 px disc; a
    hard-rounded disc would quantise the axis and flatten the boundary search.
    """
    radius = diameter_px / 2.0
    size = int(np.ceil(diameter_px)) + 2
    size += 1 - size % 2
    centre = (size - 1) / 2.0

    ss = 4
    grid = (np.arange(size * ss) + 0.5) / ss - 0.5
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    mask = ((yy - centre) ** 2 + (xx - centre) ** 2) <= radius**2
    psf = mask.astype(np.float64).reshape(size, ss, size, ss).mean(axis=(1, 3))

    total = psf.sum()
    if total <= 0.0:
        psf[:] = 0.0
        psf[int(centre), int(centre)] = 1.0
        return psf
    return psf / total


@register
class Defocus(Degradation):
    name = "defocus"
    Params = DefocusParams
    stochastic = False
    # Both axes sweep the near side of the focus distance. Beyond the focus
    # point a short lens has so much depth of field that the circle of
    # confusion never reaches one pixel -- a real optical fact, and the reason
    # a far-side sweep cannot produce a boundary to find.
    axes = (
        Axis(
            "subject_distance_m",
            "m",
            0.6,
            6.0,
            expect=("laplacian_var", "increasing"),
        ),
        Axis("aperture_f_number", "f-number", 1.4, 8.0, expect=("laplacian_var", "increasing")),
    )

    def derived(self, params: DefocusParams) -> dict[str, tuple[float, str]]:
        f = params.focal_length_mm
        s1 = params.focus_distance_m * 1000.0
        s2 = params.subject_distance_m * 1000.0
        n = params.aperture_f_number

        if s1 <= f:
            coc_mm = 0.0
        else:
            coc_mm = abs(s2 - s1) / s2 * (f * f) / (n * (s1 - f))

        coc_px = coc_mm * 1000.0 / params.pixel_pitch_um
        return {
            "coc_diameter_px": (float(coc_px), "px"),
            "coc_diameter_um": (float(coc_mm * 1000.0), "um"),
        }

    def _apply(self, image, params: DefocusParams, rng) -> np.ndarray:
        diameter = self.derived(params)["coc_diameter_px"][0]
        if diameter < 1.0:
            # A circle of confusion below one pixel is, by definition, in focus.
            return image.copy()
        psf = _disc_psf(diameter).astype(np.float32)
        return cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT_101)
