"""Detection metrics, checked against cases whose answer is known by hand.

mAP is where evaluation code quietly goes wrong: off-by-one in the IoU, a
sort that is not stable, interpolation that silently differs from the standard.
Every case here has an answer worked out independently of the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from blindspot.metrics import Detection, GroundTruth, average_precision, iou_xyxy, mean_ap50


def det(x0, y0, x1, y1, score=0.9, label=0, image="a"):
    return Detection(image_id=image, label=label, score=score, box=(x0, y0, x1, y1))


def gt(x0, y0, x1, y1, label=0, image="a"):
    return GroundTruth(image_id=image, label=label, box=(x0, y0, x1, y1))


# ---------------------------------------------------------------- IoU


def test_iou_identical_boxes_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_touching_edges_is_zero():
    """Boxes sharing only an edge have zero area of intersection."""
    assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == pytest.approx(0.0)


def test_iou_half_overlap_worked_by_hand():
    # A = [0,0,10,10] area 100; B = [5,0,15,10] area 100
    # intersection = 5*10 = 50; union = 100+100-50 = 150; IoU = 1/3
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


def test_iou_contained_box():
    # A = 100, B = 25 fully inside; union = 100; IoU = 0.25
    assert iou_xyxy((0, 0, 10, 10), (2, 2, 7, 7)) == pytest.approx(0.25)


def test_iou_is_symmetric():
    a, b = (3, 4, 19, 22), (7, 1, 25, 18)
    assert iou_xyxy(a, b) == pytest.approx(iou_xyxy(b, a))


def test_iou_of_degenerate_box_is_zero():
    assert iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------- AP


def test_average_precision_of_perfect_ranking_is_one():
    # 2 ground truths, 2 correct detections ranked first.
    assert average_precision([1, 1], n_ground_truth=2) == pytest.approx(1.0)


def test_average_precision_with_no_ground_truth_is_zero():
    assert average_precision([1, 1], n_ground_truth=0) == pytest.approx(0.0)


def test_average_precision_with_no_detections_is_zero():
    assert average_precision([], n_ground_truth=3) == pytest.approx(0.0)


def test_average_precision_missing_half_the_objects():
    # 2 GT, one correct detection: recall tops out at 0.5 with precision 1.0.
    assert average_precision([1], n_ground_truth=2) == pytest.approx(0.5)


def test_average_precision_worked_by_hand():
    """tp/fp sequence [1, 0, 1] with 2 ground truths.

    rank 1: tp=1 fp=0 -> precision 1/1 = 1.000, recall 1/2 = 0.5
    rank 2: tp=1 fp=1 -> precision 1/2 = 0.500, recall 0.5
    rank 3: tp=2 fp=1 -> precision 2/3 = 0.667, recall 1.0

    With all-point interpolation, precision is made monotonically decreasing
    from the right: [1.000, 0.667, 0.667]. AP is the area under the
    precision-recall curve = sum over recall increments:
        (0.5 - 0.0) * 1.000 + (1.0 - 0.5) * 0.667 = 0.8333
    """
    assert average_precision([1, 0, 1], n_ground_truth=2) == pytest.approx(0.8333, abs=1e-4)


# ---------------------------------------------------------------- mAP@50


def test_map50_perfect_prediction_is_one():
    truths = [gt(0, 0, 10, 10), gt(50, 50, 60, 60)]
    preds = [det(0, 0, 10, 10), det(50, 50, 60, 60)]
    assert mean_ap50(preds, truths)["mAP50"] == pytest.approx(1.0)


def test_map50_no_predictions_is_zero():
    assert mean_ap50([], [gt(0, 0, 10, 10)])["mAP50"] == pytest.approx(0.0)


def test_map50_no_ground_truth_is_zero():
    assert mean_ap50([det(0, 0, 10, 10)], [])["mAP50"] == pytest.approx(0.0)


def test_map50_threshold_is_exactly_half():
    """A box at IoU just under 0.5 is a miss; just over is a hit.

    GT [0,0,10,10] area 100. Prediction [0,0,10,W] has intersection 10W and
    union 100 + 10W - 10W... so use a shifted box instead:
    pred [dx,0,10+dx,10] -> intersection 10*(10-dx), union 200 - 10*(10-dx).
    dx=3.34 -> inter 66.6, union 133.4, IoU 0.4993  (miss)
    dx=3.33 -> inter 66.7, union 133.3, IoU 0.5004  (hit)
    """
    truths = [gt(0, 0, 10, 10)]
    assert mean_ap50([det(3.34, 0, 13.34, 10)], truths)["mAP50"] == pytest.approx(0.0)
    assert mean_ap50([det(3.33, 0, 13.33, 10)], truths)["mAP50"] == pytest.approx(1.0)


def test_map50_duplicate_detections_count_as_false_positives():
    """Two predictions on one object: the second is a false positive.

    tp/fp = [1, 0] with 1 ground truth -> AP = 1.0 * (1.0 - 0.0) = 1.0 at
    recall 1, but precision at rank 2 is 0.5. All-point interpolation keeps
    the max precision to the right, so AP stays 1.0. The duplicate is still
    reported in the false-positive count.
    """
    truths = [gt(0, 0, 10, 10)]
    preds = [det(0, 0, 10, 10, score=0.9), det(0, 0, 10, 10, score=0.8)]
    result = mean_ap50(preds, truths)
    assert result["false_positives"] == 1
    assert result["true_positives"] == 1


def test_map50_wrong_label_is_not_a_match():
    truths = [gt(0, 0, 10, 10, label=0)]
    preds = [det(0, 0, 10, 10, label=1)]
    assert mean_ap50(preds, truths)["mAP50"] == pytest.approx(0.0)


def test_map50_detection_on_another_image_is_not_a_match():
    truths = [gt(0, 0, 10, 10, image="a")]
    preds = [det(0, 0, 10, 10, image="b")]
    assert mean_ap50(preds, truths)["mAP50"] == pytest.approx(0.0)


def test_map50_averages_over_classes_not_over_instances():
    """One class with many objects must not outweigh a class with one.

    Class 0: 1 GT, detected      -> AP 1.0
    Class 1: 2 GT, none detected -> AP 0.0
    mAP = (1.0 + 0.0) / 2 = 0.5, not 1/3.
    """
    truths = [
        gt(0, 0, 10, 10, label=0),
        gt(20, 20, 30, 30, label=1),
        gt(40, 40, 50, 50, label=1),
    ]
    preds = [det(0, 0, 10, 10, label=0)]
    assert mean_ap50(preds, truths)["mAP50"] == pytest.approx(0.5)


def test_map50_higher_scoring_detection_claims_the_match():
    """Greedy matching goes in score order, which is what makes AP well-defined."""
    truths = [gt(0, 0, 10, 10)]
    good = det(0, 0, 10, 10, score=0.6)
    poor = det(1, 1, 11, 11, score=0.9)  # IoU ~0.68, still over threshold
    result = mean_ap50([good, poor], truths)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1


def test_map50_is_deterministic_under_input_order():
    """Shuffling the inputs must not move the number (rule T2 applies here too)."""
    rng = np.random.default_rng(0)
    truths = [gt(i * 10, 0, i * 10 + 8, 8, label=i % 3) for i in range(12)]
    preds = [
        det(i * 10 + 1, 1, i * 10 + 9, 9, score=float(rng.random()), label=i % 3)
        for i in range(12)
    ]
    reference = mean_ap50(preds, truths)["mAP50"]
    for _ in range(5):
        p = list(preds)
        t = list(truths)
        rng.shuffle(p)
        rng.shuffle(t)
        assert mean_ap50(p, t)["mAP50"] == pytest.approx(reference)


def test_map50_reports_the_counts_it_used():
    truths = [gt(0, 0, 10, 10), gt(50, 50, 60, 60)]
    preds = [det(0, 0, 10, 10)]
    result = mean_ap50(preds, truths)
    assert result["n_ground_truth"] == 2
    assert result["n_predictions"] == 1
    assert result["false_negatives"] == 1
