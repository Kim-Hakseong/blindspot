"""Sim-to-real: synthetic low-light prediction against real low-light photos.

No first-party capture was made. The real images come from a public
third-party dataset (ExDark, see DATASETS.md), and their scene illuminance is
*estimated, not measured*, by one of two methods, always labelled:

- EXIF: the incident-light exposure equation E = C * N^2 / (t * S) with
  C = 250, from the aperture N, exposure time t (s) and ISO S the camera
  recorded. It assumes the camera exposed the scene to mid-grey, which a
  deliberately dark shot violates, so a second variant scales it by the
  photo's mean linear luminance over 0.18.
- Image statistics: the photo's SNR inverted through Blindspot's own
  population sweep on the synthetic low-light axis (boundary/coverage.py). It
  assumes the real camera's noise behaves like the synthetic model.

The comparison is per bin of estimated illuminance: the detector's real mAP@50
on the bin's images against the synthetic mAP the low-light sweep predicts at
the bin's median estimate, both judged against the same failure threshold.
"""

from __future__ import annotations

import numpy as np

EXIF_METHOD = ("estimated, not measured: E = 250 * N^2 / (t * ISO) from EXIF aperture, "
               "exposure time and ISO (incident-light exposure equation)")
STATS_METHOD = ("estimated, not measured: image SNR inverted through Blindspot's synthetic "
                "low-light population sweep")
INCIDENT_CALIBRATION = 250.0
MID_GREY = 0.18
EXIF_CORRECTED_METHOD = ("estimated, not measured: E = 250 * N^2 / (t * ISO) from EXIF, scaled by the "
                         "photo's mean linear luminance / 0.18 (the equation assumes exposure to mid-grey)")


def parse_exdark(text: str) -> list[dict]:
    """ExDark bbGt annotation -> [{cls, box: (x0, y0, x1, y1)}]."""
    boxes = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        cls, x, y, w, h = line.split()[:5]
        x, y, w, h = map(float, (x, y, w, h))
        boxes.append({"cls": cls, "box": (x, y, x + w, y + h)})
    return boxes


def _area(box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def person_is_main_subject(boxes: list[dict]) -> bool:
    """Rule C5: exclude an image whose largest annotated object is a person."""
    if not boxes:
        return False
    return max(boxes, key=lambda b: _area(b["box"]))["cls"] == "People"


def exif_illuminance_lux(exposure_s, f_number, iso, mean_linear: float | None = None) -> float | None:
    """Incident-light exposure equation, optionally corrected for the photo's
    actual brightness: the equation assumes an exposure to mid-grey (0.18
    linear), so a darker-than-mid-grey photo implies proportionally less light."""
    if not exposure_s or not f_number or not iso:
        return None
    e = INCIDENT_CALIBRATION * float(f_number) ** 2 / (float(exposure_s) * float(iso))
    return e if mean_linear is None else e * float(mean_linear) / MID_GREY


def compare_bins(bins: list[dict], curve: list[dict], threshold_map50: float) -> dict:
    """Real vs synthetic mAP per illuminance bin, against one threshold."""
    values = np.array([p["value"] for p in curve], dtype=float)
    maps = np.array([p["map50"] for p in curve], dtype=float)
    order = np.argsort(values)
    out = []
    for b in bins:
        curve_map = float(np.interp(b["lux_median"], values[order], maps[order]))
        # A matched prediction (same exposure time as the real photos) wins over
        # the default-exposure curve when the caller has computed one.
        synthetic = b.get("synthetic_map50", curve_map)
        out.append(b | {"synthetic_map50": synthetic, "curve_map50": curve_map,
                        "gap_map50": b["real_map50"] - synthetic,
                        "synthetic_failed": synthetic < threshold_map50,
                        "real_failed": b["real_map50"] < threshold_map50})
    agree = sum(b["synthetic_failed"] == b["real_failed"] for b in out)
    return {"bins": out, "agree": agree, "disagree": len(out) - agree,
            "threshold_map50": threshold_map50}


def gap_summary(compared: dict, synthetic_boundary_lux: tuple[float, float]) -> dict:
    """The sim-to-real gap in illuminance, or a bound on it.

    If no real bin fails, the real failure point lies below the darkest bin and
    only a lower bound on how far the synthetic boundary overstates it exists:
    the boundary's lower edge over the darkest bin's median estimate.
    """
    bins = sorted(compared["bins"], key=lambda b: b["lux_median"])
    real_fail = [b for b in bins if b["real_failed"]]
    s_only = sum(b["synthetic_failed"] and not b["real_failed"] for b in bins)
    r_only = sum(b["real_failed"] and not b["synthetic_failed"] for b in bins)
    out = {"real_failure_observed": bool(real_fail),
           "darkest_bin_lux_median": bins[0]["lux_median"],
           "darkest_bin_real_map50": bins[0]["real_map50"],
           "synthetic_boundary_lux": list(synthetic_boundary_lux),
           "bins_synthetic_fail_real_pass": s_only, "bins_real_fail_synthetic_pass": r_only,
           "direction": ("synthetic predicts failure where real photos pass" if s_only > r_only else
                         "real photos fail where synthetic predicts a pass" if r_only > s_only else
                         "no net direction")}
    if not real_fail:
        out["synthetic_boundary_overstates_failure_illuminance_by_at_least"] = (
            synthetic_boundary_lux[0] / bins[0]["lux_median"])
    else:
        out["brightest_real_failing_bin_lux_median"] = max(b["lux_median"] for b in real_fail)
    return out
