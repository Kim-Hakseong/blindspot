# Degradation axes

Seven OpenCV 5 kernels, each parameterised by quantities that appear on a
camera or codec spec sheet. Every figure below is cited to
`bench/out/degrade_response.json` and verified by
`uv run python tools/check_doc_numbers.py`.

Regenerate with:

```bash
uv run python bench/degrade_response.py --out bench/out/degrade_response.json
```

Measured on OpenCV 5.0.0, seed 20260906, against the deterministic reference
scene (SHA-256 `115349bf…`), whose undegraded Laplacian variance is
2314.91 <!--bench:degrade_response.baseline_measurements.laplacian_var--> and
RMS contrast 64.19 <!--bench:degrade_response.baseline_measurements.rms_contrast-->.

## Motion blur — exposure time and angular velocity

A camera rotating at ω during an exposure of *t* sweeps an angle ω·*t*, which
displaces a scene point across the sensor by `s = f_px · tan(ω·t)`. That
displacement is the support of the PSF, built as a sub-pixel-accurate line so
that a 9.43 px smear differs from a 9 px smear.

| Parameter | Unit | Range |
|---|---|---|
| `exposure_ms` | ms | 0 – 40 |
| `angular_velocity_deg_s` | deg/s | 0 – 120 |
| `focal_length_px` | px | context |
| `direction_deg` | deg | context |

At 40 ms and 60 deg/s with a 900 px focal length the derived PSF length is
37.72 px <!--bench:degrade_response.axes[7].points[8].derived.psf_length_px.value-->,
and Laplacian variance falls from the undegraded baseline to
125.17 <!--bench:degrade_response.axes[7].points[8].measured.laplacian_var--> — an
18-fold loss of high-frequency detail.

## Defocus — thin-lens circle of confusion

`CoC = |S₂−S₁|/S₂ · f²/(N·(S₁−f))`, divided by pixel pitch to get pixels. The
PSF is a uniform disc, not a Gaussian: a disc has zeros in its transfer
function, so it removes specific spatial frequencies outright the way real
defocused optics do.

| Parameter | Unit | Range |
|---|---|---|
| `subject_distance_m` | m | 0.6 – 6.0 |
| `aperture_f_number` | f-number | 1.4 – 8.0 |
| `focal_length_mm`, `focus_distance_m`, `pixel_pitch_um` | mm, m, µm | context |

Both axes sweep the **near** side of the focus distance. This is not a choice
of convenience: an 8 mm lens at f/8.7 focused at 5 m keeps the circle of
confusion below one pixel everywhere beyond 6 m, so a far-side sweep has no
boundary to find. At 0.6 m the derived circle of confusion is
11.01 px <!--bench:degrade_response.axes[0].points[0].derived.coc_diameter_px.value-->.

## Lens distortion — Brown-Conrady via `remap`

The same `k1, k2, p1, p2` OpenCV's own calibration reports, so a user can feed
in the distortion vector they already have. Verified by a forward/inverse
roundtrip required to stay under 1 px; a warp that disagrees with its own
inverse would report our numerical error as the pipeline's tolerance.

## Rolling shutter — readout time and angular velocity

Row *r* is exposed at `t(r) = (r/H)·T_readout` and displaced by
`f_px · tan(ω·t(r))`. Applied as a per-row `remap` rather than a global affine
shear, so the model stays exact if the row-time relationship is made
non-linear later. Note that the displacement is therefore only *near*-linear in
readout time — the test pins the closed form and the direction of departure
rather than asserting linearity.

## Fog — Koschmieder atmospheric scattering

`I = J·t + A·(1−t)` with `t = exp(−β·d)`. β is the extinction coefficient in
1/m, from which meteorological visibility follows as ≈ 3.912/β. A fog boundary
in β therefore converts directly into "this pipeline fails below N metres of
visibility".

At β = 0.12 /m — a visibility of
32.6 m <!--bench:degrade_response.axes[2].points[8].derived.meteorological_visibility_m.value--> —
RMS contrast collapses to
15.37 <!--bench:degrade_response.axes[2].points[8].measured.rms_contrast-->.

**Limitation:** depth is approximated as a linear ramp increasing towards the
top of the frame, the standard ground-plane camera assumption. Scenes that
violate it will have their fog mis-graded.

## Low light — Poisson-Gaussian sensor model

Illuminance drives a physical chain: lux → photons → electrons (Poisson) → read
noise (Gaussian) → gain → full-well clip → 8-bit quantisation. Shot noise grows
as the square root of signal, so *relative* noise grows as light falls. This is
why an additive-Gaussian "noise strength" slider mispredicts the failure point.

Measured SNR spans
2.81 dB <!--bench:degrade_response.axes[6].points[0].measured.snr_db--> at 0.5 lux
to 27.38 dB <!--bench:degrade_response.axes[6].points[8].measured.snr_db--> at 400 lux.

This is the one stochastic kernel; it is seeded and byte-reproducible.

## JPEG — libjpeg quality

OpenCV's own encoder, so the quantisation tables are the standard ones. The
axis value *is* the quality setting an upstream camera or CDN would use, and a
test asserts our output is byte-identical to a plain `cv2.imencode` at the same
quality — the axis is the real scale, not an approximation of it.

## H.264 CRF — not implemented

OpenCV 5's `VideoWriter` exposes no rate-control handle in the wheel build:
neither `OPENCV_FFMPEG_WRITER_OPTIONS=crf;N` nor
`VIDEOWRITER_PROP_QUALITY` changes the encoded output (verified — encoded size
is identical at CRF 1, 23 and 45). A CRF axis therefore cannot be implemented
through OpenCV alone, and inventing a mapping from some other quality knob to a
"CRF" label would be a fabricated unit.

The compression axis is JPEG quality for now. If a CRF axis is added it will
call an encoder directly and be documented as the one place the degradation
path leaves OpenCV.

## Objective measurement

Kernels are asked for a physical condition; these report what it actually cost,
which is not the same thing:

| Measure | What it detects |
|---|---|
| `laplacian_var` | High-frequency detail — falls under blur |
| `rms_contrast` | Dynamic range — falls under scattering |
| `snr_db` | Noise, via Immerkær's estimator against mean level |
| `blockiness` | DCT block-edge energy on the 8 px grid, as a ratio to off-grid |
| `mtf50_cy_px` | Spatial frequency response (see caveat) |

**`mtf50_cy_px` is a proxy.** A true ISO 12233 MTF50 requires a slanted-edge
target in frame. For arbitrary scenes this uses the radially averaged power
spectrum normalised to DC. It is comparable between degradations *of the same
scene* and not between different scenes, and is labelled as such wherever
reported.
