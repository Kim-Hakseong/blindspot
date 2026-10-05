"""Sim-to-real on a third-party set of real low-light photos (W6-7).

No first-party capture: the real images come from a public dataset, and their
illuminance is *estimated*, never measured. These tests pin the pieces that
decide what is compared: annotation parsing, the person-as-main-subject
exclusion (rule C5), the two illuminance estimators and their labels, and the
per-bin comparison against Blindspot's synthetic prediction.
"""

from __future__ import annotations

import pytest

from blindspot.sim2real import (
    EXIF_METHOD,
    STATS_METHOD,
    compare_bins,
    exif_illuminance_lux,
    parse_exdark,
    person_is_main_subject,
)

ANNOTATION = """% bbGt version=3
People 565 217 34 102 0 0 0 0 0 0 0
Car 398 201 196 119 0 0 0 0 0 0 0
Bus 10 20 30 40 0 0 0 0 0 0 0
"""


def test_exdark_boxes_become_xyxy_with_their_class():
    boxes = parse_exdark(ANNOTATION)
    assert boxes[1] == {"cls": "Car", "box": (398.0, 201.0, 594.0, 320.0)}
    assert [b["cls"] for b in boxes] == ["People", "Car", "Bus"]


def test_an_image_whose_largest_box_is_a_person_is_excluded():
    assert not person_is_main_subject(parse_exdark(ANNOTATION))  # the car is larger
    big_person = ANNOTATION + "People 0 0 400 400 0 0 0 0 0 0 0\n"
    assert person_is_main_subject(parse_exdark(big_person))


def test_exif_illuminance_uses_the_incident_light_exposure_equation():
    # E = C * N^2 / (t * S), C = 250 (incident-meter calibration constant):
    # f/2, 1/30 s, ISO 800 -> 250 * 4 / (0.0333 * 800) = 37.5 lux.
    assert exif_illuminance_lux(1 / 30, 2.0, 800) == pytest.approx(37.5)
    assert exif_illuminance_lux(None, 2.0, 800) is None  # missing EXIF: no estimate
    assert "estimated, not measured" in EXIF_METHOD and "estimated, not measured" in STATS_METHOD


def test_bins_compare_real_and_synthetic_against_the_same_threshold():
    # Synthetic curve: mAP 0.6 at 100 lux falling to 0.1 at 5 lux.
    curve = [{"value": 100.0, "map50": 0.6}, {"value": 20.0, "map50": 0.4},
             {"value": 5.0, "map50": 0.1}]
    bins = [{"lux_median": 100.0, "real_map50": 0.55},   # both pass
            {"lux_median": 20.0, "real_map50": 0.20},    # synthetic pass, real fail
            {"lux_median": 5.0, "real_map50": 0.05}]     # both fail
    out = compare_bins(bins, curve, threshold_map50=0.3)
    assert [b["synthetic_failed"] for b in out["bins"]] == [False, False, True]
    assert [b["real_failed"] for b in out["bins"]] == [False, True, True]
    assert out["agree"] == 2 and out["disagree"] == 1
    assert out["bins"][1]["synthetic_map50"] == pytest.approx(0.4)
    assert out["bins"][1]["gap_map50"] == pytest.approx(0.20 - 0.4)


def test_a_bin_can_carry_its_own_matched_synthetic_prediction():
    # The synthetic prediction is run at the bin's estimated illuminance AND the
    # cameras' recorded exposure time; the default-exposure curve is a fallback.
    curve = [{"value": 100.0, "map50": 0.6}, {"value": 5.0, "map50": 0.1}]
    bins = [{"lux_median": 5.0, "real_map50": 0.5, "synthetic_map50": 0.45}]
    out = compare_bins(bins, curve, threshold_map50=0.3)
    b = out["bins"][0]
    assert b["synthetic_map50"] == 0.45 and not b["synthetic_failed"]
    assert b["curve_map50"] == pytest.approx(0.1)  # what the default-exposure curve says
    assert b["gap_map50"] == pytest.approx(0.05)


def test_exif_estimate_can_be_corrected_for_a_deliberately_dark_exposure():
    # The exposure equation assumes the camera exposed to mid-grey (0.18 linear).
    # A photo that came out at 0.045 linear mean is a quarter as bright, so the
    # scene was a quarter as lit as the uncorrected equation says.
    base = exif_illuminance_lux(1 / 30, 2.0, 800)
    assert exif_illuminance_lux(1 / 30, 2.0, 800, mean_linear=0.045) == pytest.approx(base / 4)
    assert exif_illuminance_lux(1 / 30, 2.0, 800, mean_linear=0.18) == pytest.approx(base)


def test_the_gap_states_the_direction_and_a_bound_when_real_never_fails():
    # Real photos never cross the threshold, even in the darkest bin: the real
    # failure point is not observed, so only a bound on the gap can be given.
    from blindspot.sim2real import gap_summary
    compared = compare_bins(
        [{"lux_median": 1.0, "real_map50": 0.40, "synthetic_map50": 0.0},
         {"lux_median": 30.0, "real_map50": 0.65, "synthetic_map50": 0.45}],
        [{"value": 1.0, "map50": 0.0}], threshold_map50=0.367)
    g = gap_summary(compared, synthetic_boundary_lux=(13.0, 25.0))
    assert g["real_failure_observed"] is False
    assert g["direction"] == "synthetic predicts failure where real photos pass"
    assert g["darkest_bin_lux_median"] == 1.0
    assert g["synthetic_boundary_overstates_failure_illuminance_by_at_least"] == pytest.approx(13.0)
    assert g["bins_synthetic_fail_real_pass"] == 1 and g["bins_real_fail_synthetic_pass"] == 0
