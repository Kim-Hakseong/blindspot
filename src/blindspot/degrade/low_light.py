"""Low light: a Poisson-Gaussian sensor model parameterised by illuminance.

The dominant noise in a real image sensor is not additive Gaussian. It is shot
noise: photon arrivals are Poisson, so the noise standard deviation grows as
the square root of the signal and the *relative* noise grows as illuminance
falls. That is precisely why a detector degrades the way it does at dusk, and
why an additive-Gaussian "noise strength" slider mispredicts the failure point.

The chain modelled here is the one a sensor datasheet describes:

    lux  ->  photons/px  ->  electrons (Poisson)  ->  + read noise (Gaussian)
         ->  gain  ->  clip at full well  ->  quantise to 8 bit

``photons_per_lux_s`` folds pixel area, lens transmission and quantum
efficiency into a single calibration constant, which is the one number a user
would tune to match their own camera.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import Field

from .base import Axis, DegradeParams, Degradation, register


class LowLightParams(DegradeParams):
    illuminance_lux: float = Field(gt=0.0, le=100000.0)
    exposure_ms: float = Field(default=10.0, gt=0.0, le=1000.0)
    iso_gain: float = Field(default=4.0, gt=0.0, le=256.0)
    read_noise_e: float = Field(default=2.0, ge=0.0, le=100.0)
    full_well_e: float = Field(default=6000.0, gt=0.0, le=200000.0)
    # Sensor calibration: electrons per lux-second at unit scene reflectance.
    photons_per_lux_s: float = Field(default=180.0, gt=0.0, le=100000.0)


@register
class LowLight(Degradation):
    name = "low_light"
    Params = LowLightParams
    stochastic = True
    axes = (Axis(
            "illuminance_lux",
            "lux",
            0.5,
            400.0,
            expect=("snr_db", "increasing"),
            # Darkness is the harsh condition: severity is at the low end.
            severe_end="lo",
        ),)

    def derived(self, params: LowLightParams) -> dict[str, tuple[float, str]]:
        # Electrons collected by a mid-grey (reflectance 0.5) patch.
        signal_e = (
            params.illuminance_lux
            * (params.exposure_ms / 1000.0)
            * params.photons_per_lux_s
        )
        shot = np.sqrt(signal_e * 0.5)
        total_noise = float(np.sqrt(shot**2 + params.read_noise_e**2))
        snr = 20.0 * np.log10(max(signal_e * 0.5, 1e-9) / max(total_noise, 1e-9))
        return {
            "signal_e": (float(signal_e), "dimensionless"),
            "shot_noise_e": (float(shot), "dimensionless"),
            "modelled_snr_db": (float(snr), "dimensionless"),
        }

    def _apply(self, image, params: LowLightParams, rng) -> np.ndarray:
        # Scene reflectance in [0, 1]; linearise out of the 8-bit sRGB-ish encode.
        reflectance = (image.astype(np.float32) / 255.0) ** 2.2

        # Photons -> electrons, per pixel, scaled by local reflectance.
        full_scale_e = (
            params.illuminance_lux * (params.exposure_ms / 1000.0) * params.photons_per_lux_s
        )
        electrons = reflectance * full_scale_e

        # Shot noise: Poisson in the electron domain.
        noisy = rng.poisson(np.clip(electrons, 0.0, None)).astype(np.float32)

        # Read noise: additive Gaussian, in electrons.
        if params.read_noise_e > 0.0:
            noisy += rng.normal(0.0, params.read_noise_e, size=noisy.shape).astype(np.float32)

        # Analogue gain, clip at the full well, then re-encode to 8 bit.
        signal = np.clip(noisy * params.iso_gain, 0.0, params.full_well_e)
        normalised = signal / params.full_well_e
        encoded = np.clip(normalised, 0.0, 1.0) ** (1.0 / 2.2)
        return np.clip(encoded * 255.0, 0, 255).astype(np.uint8)
