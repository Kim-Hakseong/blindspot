"""Objective measurement of how degraded a frame actually is.

A degradation kernel is asked for 20 ms of exposure; these functions report
what that did to the image, independently of what was asked for. The two
numbers are not the same thing, and keeping them separate is what lets the
report say "you requested this condition, here is what it objectively cost you"
rather than assuming the model was obeyed.

Everything here is deterministic and free of any model, by construction: this
module sits on the judgment path and is covered by
``tests/test_no_llm_in_judgment.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "measure",
    "laplacian_var",
    "rms_contrast",
    "snr_db",
    "blockiness",
    "mtf50_cy_px",
    "estimate_noise_sigma",
]


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def laplacian_var(image: np.ndarray) -> float:
    """Variance of the Laplacian -- the standard focus/sharpness proxy.

    Falls as the image is blurred. Sensitive to noise, which is why the low
    light axis is judged on SNR rather than on this.
    """
    # OpenCV 5 has no float32 -> float64 Laplacian path; compute in float32 and
    # take the variance in float64.
    return float(cv2.Laplacian(_gray(image), cv2.CV_32F).astype(np.float64).var())


def rms_contrast(image: np.ndarray) -> float:
    """RMS contrast: the standard deviation of luminance, in grey levels.

    This is what atmospheric scattering destroys -- fog does not blur, it
    compresses the dynamic range towards the airlight.
    """
    return float(_gray(image).std())


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Immerkaer's fast noise estimate: sigma from a Laplacian-like kernel.

    The 3x3 mask is orthogonal to first- and second-order image structure, so
    its response over a natural image is dominated by noise rather than by
    scene content.
    """
    g = _gray(image)
    h, w = g.shape
    mask = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], np.float32)
    response = np.abs(cv2.filter2D(g, cv2.CV_32F, mask, borderType=cv2.BORDER_REFLECT_101))
    # Trim the border, where the reflected padding biases the response.
    interior = response[2:-2, 2:-2] if h > 8 and w > 8 else response
    return float(interior.mean() * np.sqrt(np.pi / 2.0) / 6.0)


def snr_db(image: np.ndarray) -> float:
    """Signal-to-noise ratio in dB, as 20*log10(mean signal / noise sigma).

    Mean level rather than signal variance is used as the numerator so that a
    darker, noisier frame scores lower even when its scene content is identical
    -- which is the behaviour the illuminance axis needs to be judged on.
    """
    g = _gray(image)
    sigma = estimate_noise_sigma(image)
    return float(20.0 * np.log10(max(g.mean(), 1e-6) / max(sigma, 1e-6)))


def blockiness(image: np.ndarray) -> float:
    """DCT block-edge energy: discontinuity on the 8-px grid versus off it.

    A ratio rather than an absolute difference, so it measures the *artefact*
    and not merely how much edge content the scene happens to contain.
    """
    g = _gray(image)
    h, w = g.shape

    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))

    # Columns/rows lying on the 8-pixel JPEG block boundary.
    on_cols = np.arange(7, w - 1, 8)
    on_rows = np.arange(7, h - 1, 8)
    if on_cols.size == 0 or on_rows.size == 0:
        return 0.0

    off_cols = np.setdiff1d(np.arange(dh.shape[1]), on_cols)
    off_rows = np.setdiff1d(np.arange(dv.shape[0]), on_rows)
    if off_cols.size == 0 or off_rows.size == 0:
        return 0.0

    on = (dh[:, on_cols].mean() + dv[on_rows, :].mean()) / 2.0
    off = (dh[:, off_cols].mean() + dv[off_rows, :].mean()) / 2.0
    return float(on / max(off, 1e-6))


def mtf50_cy_px(image: np.ndarray) -> float:
    """MTF50 estimate in cycles/pixel from the radially averaged spectrum.

    A true ISO 12233 MTF50 needs a slanted-edge target in the frame. For an
    arbitrary scene this uses the radially averaged power spectrum normalised
    to DC and reports where it falls to half. It is a *proxy*, comparable
    between degradations of the same scene but not between different scenes,
    and it is labelled as such wherever it is reported.
    """
    g = _gray(image)
    g = g - g.mean()
    if g.size == 0 or not np.any(g):
        return 0.0

    window = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(g * window)))

    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    n_bins = min(cy, cx)
    if n_bins < 2:
        return 0.0
    radial = np.bincount(r.ravel(), spectrum.ravel(), minlength=n_bins)[:n_bins]
    counts = np.bincount(r.ravel(), minlength=n_bins)[:n_bins]
    profile = radial / np.maximum(counts, 1)

    if profile[0] <= 0:
        return 0.0
    normalised = profile / profile[0]

    below = np.nonzero(normalised < 0.5)[0]
    if below.size == 0:
        return 0.5  # Nyquist: never fell to half.
    i = int(below[0])
    if i == 0:
        return 0.0

    # Linear interpolation between the bracketing bins.
    y0, y1 = normalised[i - 1], normalised[i]
    frac = (y0 - 0.5) / max(y0 - y1, 1e-12)
    bin_index = (i - 1) + frac
    # Bin index -> cycles/pixel: the profile spans DC to Nyquist (0.5 cy/px).
    return float(0.5 * bin_index / n_bins)


def measure(image: np.ndarray) -> dict[str, float]:
    """All objective degradation measurements for one frame."""
    return {
        "laplacian_var": laplacian_var(image),
        "rms_contrast": rms_contrast(image),
        "snr_db": snr_db(image),
        "blockiness": blockiness(image),
        "mtf50_cy_px": mtf50_cy_px(image),
        "mean_level": float(_gray(image).mean()),
    }
