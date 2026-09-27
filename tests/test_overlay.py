"""Failure-frame overlays (W2-7).

The overlay colours a predicted box by whether it actually matched ground
truth. That verdict must come from the same matching rule that produces mAP,
or the picture and the number could disagree -- so the tests pin agreement
with `mean_ap50`, then check the drawing at known pixel positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.metrics import Detection, GroundTruth, match_detections, mean_ap50
from blindspot.overlay import CAPTION_HEIGHT, COLORS_BGR, draw_overlay
from blindspot.privacy import blur_regions


def det(x0, y0, x1, y1, score=0.9, label=0, image="a"):
    return Detection(image_id=image, label=label, score=score, box=(x0, y0, x1, y1))


def gt(x0, y0, x1, y1, label=0, image="a"):
    return GroundTruth(image_id=image, label=label, box=(x0, y0, x1, y1))


# ------------------------------------------------------------ matching


def test_match_marks_true_and_false_positives():
    truths = [gt(10, 10, 50, 50)]
    preds = [det(10, 10, 50, 50, score=0.9), det(100, 100, 140, 140, score=0.95)]
    matches = match_detections(preds, truths)
    by_box = {m.detection.box: m for m in matches}
    assert by_box[(10, 10, 50, 50)].matched is True
    assert by_box[(100, 100, 140, 140)].matched is False


def test_match_counts_agree_with_map50():
    rng = np.random.default_rng(3)
    truths = [gt(i * 30, 0, i * 30 + 20, 20, label=i % 2) for i in range(8)]
    preds = [
        det(i * 30 + float(rng.integers(-8, 8)), 0, i * 30 + 20, 20,
            score=float(rng.random()), label=i % 2)
        for i in range(8)
    ] + [det(500, 500, 520, 520, score=0.99)]
    matches = match_detections(preds, truths)
    scored = mean_ap50(preds, truths)
    assert sum(m.matched for m in matches) == scored["true_positives"]
    assert sum(not m.matched for m in matches) == scored["false_positives"]


def test_match_reports_iou_of_the_matched_truth():
    matches = match_detections([det(0, 0, 10, 10)], [gt(5, 0, 15, 10)])
    assert matches[0].matched is False  # IoU 1/3 < 0.5
    assert matches[0].best_iou == pytest.approx(1 / 3)


# ------------------------------------------------------------ drawing


def blank(h=200, w=300):
    return np.full((h, w, 3), 30, np.uint8)


def test_overlay_does_not_mutate_input():
    img = blank()
    before = img.tobytes()
    draw_overlay(img, match_detections([det(10, 10, 60, 60)], [gt(10, 10, 60, 60)]),
                 [gt(10, 10, 60, 60)], caption="x")
    assert img.tobytes() == before


def test_true_positive_is_drawn_in_pass_colour():
    truths = [gt(40, 60, 120, 140)]
    out = draw_overlay(blank(), match_detections([det(40, 60, 120, 140)], truths),
                       truths, caption="")
    # Middle of the box's left edge.
    assert tuple(out[CAPTION_HEIGHT + 100, 40]) == COLORS_BGR["pass"]


def test_false_positive_is_drawn_in_fail_colour():
    out = draw_overlay(blank(), match_detections([det(150, 60, 250, 140)], []),
                       [], caption="")
    assert tuple(out[CAPTION_HEIGHT + 100, 150]) == COLORS_BGR["fail"]


def test_ground_truth_is_dashed_not_solid():
    truths = [gt(20, 20, 280, 180)]
    out = draw_overlay(blank(), [], truths, caption="")
    top_edge = out[CAPTION_HEIGHT + 20, 30:270]
    drawn = np.any(top_edge != 30, axis=1)
    assert drawn.any() and not drawn.all(), "ground truth edge should be dashed"


def test_caption_bar_is_added_above_the_frame():
    img = blank()
    out = draw_overlay(img, [], [], caption="motion_blur 12.5 ms")
    assert out.shape[1] == img.shape[1]
    assert out.shape[0] > img.shape[0]


def test_confidence_is_written_next_to_each_prediction():
    """The silent failure is a confident wrong box; the confidence must show."""
    preds = match_detections([det(100, 80, 200, 160, score=0.91)], [])
    with_label = draw_overlay(blank(), preds, [], caption="")
    no_label = draw_overlay(blank(), preds, [], caption="", show_scores=False)
    assert with_label.tobytes() != no_label.tobytes()


# ------------------------------------------------------------ privacy


def test_blur_regions_changes_only_the_region():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_regions(img, [(20, 20, 60, 60)])
    assert out[30:50, 30:50].tobytes() != img[30:50, 30:50].tobytes()
    assert out[80:, 80:].tobytes() == img[80:, 80:].tobytes()
    assert img.tobytes() != out.tobytes()


def test_blur_regions_with_nothing_is_a_copy():
    img = blank()
    out = blur_regions(img, [])
    assert out.tobytes() == img.tobytes() and out is not img


def test_blur_regions_clips_boxes_outside_the_frame():
    img = blank(50, 50)
    out = blur_regions(img, [(-10, -10, 200, 200)])
    assert out.shape == img.shape
