"""Per-kernel behaviour: the physics, not just the monotonicity.

A kernel can move a measurement the right way and still be wrong. These tests
check the derived physical quantities against closed-form values and check the
geometric warps against their known inverse.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from blindspot.degrade import REGISTRY
from blindspot.degrade.defocus import Defocus
from blindspot.degrade.fog import Fog
from blindspot.degrade.jpeg import Jpeg
from blindspot.degrade.lens_distortion import LensDistortion
from blindspot.degrade.low_light import LowLight
from blindspot.degrade.motion_blur import MotionBlur
from blindspot.degrade.rolling_shutter import RollingShutter

SEED = 11


# --------------------------------------------------------------------------
# Motion blur: PSF length must follow s = f * omega * t
# --------------------------------------------------------------------------


def test_motion_blur_psf_length_matches_closed_form():
    deg = MotionBlur()
    # 30 deg/s for 20 ms at 900 px focal length:
    #   theta = 30 * 0.020 = 0.6 deg = 0.010472 rad
    #   s = 900 * tan(0.010472) = 9.4254 px
    params = deg.Params(
        exposure_ms=20.0,
        angular_velocity_deg_s=30.0,
        focal_length_px=900.0,
        direction_deg=0.0,
    )
    psf_px, unit = deg.derived(params)["psf_length_px"]
    expected = 900.0 * np.tan(np.deg2rad(30.0 * 0.020))
    assert unit == "px"
    assert psf_px == pytest.approx(expected, rel=1e-9)
    assert psf_px == pytest.approx(9.4254, abs=1e-3)


def test_motion_blur_doubling_exposure_doubles_psf_length():
    deg = MotionBlur()
    base = dict(angular_velocity_deg_s=20.0, focal_length_px=800.0, direction_deg=0.0)
    a = deg.derived(deg.Params(exposure_ms=5.0, **base))["psf_length_px"][0]
    b = deg.derived(deg.Params(exposure_ms=10.0, **base))["psf_length_px"][0]
    assert b / a == pytest.approx(2.0, rel=1e-3)


def test_motion_blur_is_directional(scene):
    """Horizontal smear must differ from vertical smear."""
    deg = MotionBlur()
    kw = dict(exposure_ms=25.0, angular_velocity_deg_s=40.0, focal_length_px=900.0)
    h = deg.apply(scene, deg.Params(direction_deg=0.0, **kw), seed=SEED)
    v = deg.apply(scene, deg.Params(direction_deg=90.0, **kw), seed=SEED)
    assert h.tobytes() != v.tobytes()

    g = lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # A horizontal smear preserves vertical detail and destroys horizontal detail.
    h_dx = np.abs(cv2.Sobel(g(h), cv2.CV_32F, 1, 0, ksize=3)).mean()
    h_dy = np.abs(cv2.Sobel(g(h), cv2.CV_32F, 0, 1, ksize=3)).mean()
    assert h_dy > h_dx


def test_motion_blur_zero_exposure_is_a_no_op(scene):
    deg = MotionBlur()
    params = deg.Params(
        exposure_ms=0.0, angular_velocity_deg_s=50.0, focal_length_px=900.0, direction_deg=0.0
    )
    assert deg.apply(scene, params, seed=SEED).tobytes() == scene.tobytes()


# --------------------------------------------------------------------------
# Defocus: circle of confusion from the thin-lens equation
# --------------------------------------------------------------------------


def test_defocus_coc_matches_thin_lens_closed_form():
    deg = Defocus()
    # f=8mm, N=2.0, focused at 5m, subject at 20m, pixel pitch 2um
    #   CoC = |S2-S1|/S2 * f^2 / (N*(S1-f))
    #       = (15/20) * 64 / (2*(5000-8)) mm   [S in mm]
    #       = 0.75 * 64 / 9984 = 0.004808 mm = 4.808 um = 2.404 px
    params = deg.Params(
        focal_length_mm=8.0,
        aperture_f_number=2.0,
        focus_distance_m=5.0,
        subject_distance_m=20.0,
        pixel_pitch_um=2.0,
    )
    d = deg.derived(params)
    coc_mm = (15.0 / 20.0) * 64.0 / (2.0 * (5000.0 - 8.0))
    assert d["coc_diameter_px"][1] == "px"
    assert d["coc_diameter_px"][0] == pytest.approx(coc_mm * 1000.0 / 2.0, rel=1e-6)
    assert d["coc_diameter_px"][0] == pytest.approx(2.404, abs=1e-3)


def test_defocus_at_the_focus_distance_is_a_no_op(scene):
    deg = Defocus()
    params = deg.Params(
        focal_length_mm=8.0,
        aperture_f_number=2.0,
        focus_distance_m=6.0,
        subject_distance_m=6.0,
        pixel_pitch_um=2.0,
    )
    assert deg.derived(params)["coc_diameter_px"][0] == pytest.approx(0.0, abs=1e-12)
    assert deg.apply(scene, params, seed=SEED).tobytes() == scene.tobytes()


def test_defocus_wider_aperture_blurs_more():
    """Smaller f-number = wider aperture = larger circle of confusion."""
    deg = Defocus()
    kw = dict(
        focal_length_mm=8.0, focus_distance_m=5.0, subject_distance_m=20.0, pixel_pitch_um=2.0
    )
    wide = deg.derived(deg.Params(aperture_f_number=1.4, **kw))["coc_diameter_px"][0]
    narrow = deg.derived(deg.Params(aperture_f_number=11.0, **kw))["coc_diameter_px"][0]
    assert wide > narrow


# --------------------------------------------------------------------------
# Lens distortion: warp must invert to within a pixel
# --------------------------------------------------------------------------


def test_lens_distortion_roundtrips_under_one_pixel():
    """Rule from the sprint plan: forward then inverse map < 1 px error."""
    deg = LensDistortion()
    params = deg.Params(k1=0.15, k2=0.02, p1=0.0005, p2=0.0005)
    err = deg.roundtrip_error_px(width=256, height=256, params=params)
    assert err < 1.0, f"distortion roundtrip error {err:.3f} px exceeds 1 px"


def test_lens_distortion_zero_coefficients_is_a_no_op(grid):
    deg = LensDistortion()
    params = deg.Params(k1=0.0, k2=0.0, p1=0.0, p2=0.0)
    out = deg.apply(grid, params, seed=SEED)
    # Bilinear resampling on an identity map is not bit-exact at the border;
    # require the interior to be unchanged within one grey level.
    diff = np.abs(out[4:-4, 4:-4].astype(np.int16) - grid[4:-4, 4:-4].astype(np.int16))
    assert diff.max() <= 1


def test_lens_distortion_bends_straight_lines(grid):
    """A barrel-distorted grid line must deviate from straight."""
    deg = LensDistortion()
    out = deg.apply(grid, deg.Params(k1=0.35, k2=0.05, p1=0.0, p2=0.0), seed=SEED)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # Track the brightest row near the top edge across columns; under barrel
    # distortion its row index must vary.
    top = g[:60, :]
    rows = np.argmax(top, axis=0)[20:-20]
    assert rows.std() > 0.5, "distortion did not bend the grid"


# --------------------------------------------------------------------------
# Rolling shutter: per-row displacement is linear in readout time
# --------------------------------------------------------------------------


def test_rolling_shutter_shift_matches_closed_form():
    deg = RollingShutter()
    kw = dict(angular_velocity_deg_s=60.0, focal_length_px=900.0, height_px=256)
    for t_ms in (10.0, 20.0, 35.0):
        got = deg.derived(deg.Params(readout_time_ms=t_ms, **kw))["row_shift_total_px"][0]
        expected = 900.0 * np.tan(np.deg2rad(60.0 * t_ms / 1000.0))
        assert got == pytest.approx(expected, rel=1e-9)


def test_rolling_shutter_shear_is_near_linear_at_small_angles():
    """Displacement is f*tan(omega*t), so it is linear only in the small-angle
    limit. Doubling readout time slightly more than doubles the shear, and the
    model must show that rather than hide it behind a linear approximation."""
    deg = RollingShutter()
    kw = dict(angular_velocity_deg_s=60.0, focal_length_px=900.0, height_px=256)
    a = deg.derived(deg.Params(readout_time_ms=10.0, **kw))["row_shift_total_px"][0]
    b = deg.derived(deg.Params(readout_time_ms=20.0, **kw))["row_shift_total_px"][0]
    ratio = b / a
    assert ratio == pytest.approx(2.0, rel=1e-3)
    assert ratio > 2.0, "tan() must grow slightly faster than linear"


def test_rolling_shutter_tilts_a_vertical_bar():
    """The signature artefact: a vertical edge leans."""
    deg = RollingShutter()
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, 120:136] = 255
    out = deg.apply(
        img,
        deg.Params(
            readout_time_ms=30.0, angular_velocity_deg_s=80.0, focal_length_px=900.0, height_px=256
        ),
        seed=SEED,
    )
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    centroid = lambda row: (np.arange(256) * g[row]).sum() / max(g[row].sum(), 1e-6)
    top, bottom = centroid(10), centroid(245)
    assert abs(bottom - top) > 2.0, "vertical bar did not shear"


def test_rolling_shutter_zero_readout_is_a_no_op(scene):
    deg = RollingShutter()
    params = deg.Params(
        readout_time_ms=0.0, angular_velocity_deg_s=80.0, focal_length_px=900.0, height_px=256
    )
    out = deg.apply(scene, params, seed=SEED)
    diff = np.abs(out.astype(np.int16) - scene.astype(np.int16))
    assert diff.max() <= 1


# --------------------------------------------------------------------------
# Fog: Koschmieder / atmospheric scattering
# --------------------------------------------------------------------------


def test_fog_transmission_follows_beer_lambert():
    deg = Fog()
    params = deg.Params(beta_per_m=0.1, depth_near_m=5.0, depth_far_m=50.0, airlight=0.85)
    d = deg.derived(params)
    # t = exp(-beta * d)
    assert d["transmission_near"][0] == pytest.approx(np.exp(-0.1 * 5.0), rel=1e-9)
    assert d["transmission_far"][0] == pytest.approx(np.exp(-0.1 * 50.0), rel=1e-9)


def test_fog_zero_beta_is_a_no_op(scene):
    deg = Fog()
    params = deg.Params(beta_per_m=0.0, depth_near_m=5.0, depth_far_m=50.0, airlight=0.85)
    out = deg.apply(scene, params, seed=SEED)
    assert np.abs(out.astype(np.int16) - scene.astype(np.int16)).max() <= 1


def test_fog_pulls_the_image_towards_airlight(scene):
    """At very high beta the scene is almost entirely airlight."""
    deg = Fog()
    params = deg.Params(beta_per_m=2.0, depth_near_m=5.0, depth_far_m=50.0, airlight=0.85)
    out = deg.apply(scene, params, seed=SEED)
    assert out.mean() == pytest.approx(0.85 * 255.0, abs=6.0)


def test_fog_is_depth_graded(scene):
    """Far parts of the frame must be foggier than near parts."""
    deg = Fog()
    out = deg.apply(
        scene,
        deg.Params(beta_per_m=0.08, depth_near_m=5.0, depth_far_m=80.0, airlight=0.9),
        seed=SEED,
    )
    g_in = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Depth increases towards the top of the frame -> contrast loss is larger there.
    top_loss = g_in[:64].std() - g_out[:64].std()
    bottom_loss = g_in[-64:].std() - g_out[-64:].std()
    assert top_loss > bottom_loss


# --------------------------------------------------------------------------
# Low light: Poisson-Gaussian sensor model
# --------------------------------------------------------------------------


def test_low_light_photon_count_scales_with_illuminance_and_exposure():
    deg = LowLight()
    kw = dict(exposure_ms=10.0, iso_gain=4.0, read_noise_e=2.0, full_well_e=6000.0)
    a = deg.derived(deg.Params(illuminance_lux=10.0, **kw))["signal_e"][0]
    b = deg.derived(deg.Params(illuminance_lux=20.0, **kw))["signal_e"][0]
    assert b / a == pytest.approx(2.0, rel=1e-6)


def test_low_light_darker_scenes_are_noisier(scene):
    """Shot noise dominates as illuminance falls -- the whole point of the axis."""
    from blindspot.measure import measure

    deg = LowLight()
    kw = dict(exposure_ms=10.0, iso_gain=4.0, read_noise_e=2.0, full_well_e=6000.0)
    bright = measure(deg.apply(scene, deg.Params(illuminance_lux=400.0, **kw), seed=SEED))
    dark = measure(deg.apply(scene, deg.Params(illuminance_lux=2.0, **kw), seed=SEED))
    assert dark["snr_db"] < bright["snr_db"]


def test_low_light_noise_is_seed_reproducible(scene):
    deg = LowLight()
    params = deg.Params(
        illuminance_lux=5.0, exposure_ms=10.0, iso_gain=4.0, read_noise_e=2.0, full_well_e=6000.0
    )
    a = deg.apply(scene, params, seed=SEED)
    b = deg.apply(scene, params, seed=SEED)
    c = deg.apply(scene, params, seed=SEED + 1)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


# --------------------------------------------------------------------------
# JPEG: quality is the standard 0-100 libjpeg scale
# --------------------------------------------------------------------------


def test_jpeg_quality_100_is_near_lossless(scene):
    deg = Jpeg()
    out = deg.apply(scene, deg.Params(quality=100.0), seed=SEED)
    assert np.abs(out.astype(np.int16) - scene.astype(np.int16)).mean() < 2.0


def test_jpeg_lower_quality_costs_more_error(scene):
    deg = Jpeg()
    err = lambda q: np.abs(
        deg.apply(scene, deg.Params(quality=q), seed=SEED).astype(np.int16)
        - scene.astype(np.int16)
    ).mean()
    assert err(10.0) > err(50.0) > err(95.0)


def test_jpeg_quality_is_the_real_libjpeg_scale(scene):
    """Our output must equal a plain OpenCV encode at the same quality."""
    deg = Jpeg()
    q = 37
    ours = deg.apply(scene, deg.Params(quality=float(q)), seed=SEED)
    ok, buf = cv2.imencode(".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    reference = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert ours.tobytes() == reference.tobytes()


# --------------------------------------------------------------------------
# Registry hygiene
# --------------------------------------------------------------------------


def test_all_expected_axes_are_registered():
    expected = {
        "motion_blur",
        "defocus",
        "lens_distortion",
        "rolling_shutter",
        "fog",
        "low_light",
        "jpeg",
    }
    assert expected <= set(REGISTRY.names())


def test_params_reject_unknown_fields():
    deg = MotionBlur()
    with pytest.raises(Exception):
        deg.Params(
            exposure_ms=5.0,
            angular_velocity_deg_s=10.0,
            focal_length_px=900.0,
            direction_deg=0.0,
            intensity=0.5,  # an arbitrary scale must not be accepted
        )


def test_params_are_immutable():
    deg = MotionBlur()
    p = deg.Params(
        exposure_ms=5.0, angular_velocity_deg_s=10.0, focal_length_px=900.0, direction_deg=0.0
    )
    with pytest.raises(Exception):
        p.exposure_ms = 99.0
