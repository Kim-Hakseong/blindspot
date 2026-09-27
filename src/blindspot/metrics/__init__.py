"""Detection metrics.

This module decides whether a pipeline passed or failed under a condition, so
it is on the judgment path: it is deterministic, it has no cloud or model
dependency, and `tests/test_no_llm_in_judgment.py` enforces that.

The mAP@50 here is the all-point-interpolated average precision used by the
COCO and Pascal VOC 2010+ conventions: greedy matching in descending score
order, one ground truth per detection, IoU >= 0.5, averaged over classes
rather than over instances so a crowded class cannot dominate the number.

Results are invariant to the order the inputs arrive in, which matters because
probes come back from parallel workers in nondeterministic order and the
reported boundary must not depend on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "Detection",
    "GroundTruth",
    "iou_xyxy",
    "average_precision",
    "mean_ap50",
    "match_detections",
    "Match",
    "IOU_THRESHOLD",
]

#: The IoU at which a detection counts as a match. Defined once, here.
IOU_THRESHOLD = 0.5

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A predicted box. ``box`` is (x0, y0, x1, y1) in pixels."""

    image_id: str
    label: int
    score: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    """A labelled box. ``box`` is (x0, y0, x1, y1) in pixels."""

    image_id: str
    label: int
    box: Box


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    intersection = inter_w * inter_h

    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def average_precision(matches: Sequence[int], n_ground_truth: int) -> float:
    """All-point interpolated AP from a score-ordered hit/miss sequence.

    ``matches[i]`` is 1 if the i-th highest-scoring detection matched an unused
    ground truth, else 0. Precision is made monotonically non-increasing from
    the right before integrating, which is the standard interpolation and the
    thing most hand-rolled implementations get wrong.
    """
    if n_ground_truth <= 0 or not matches:
        return 0.0

    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for matched in matches:
        if matched:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_ground_truth)

    # Interpolate: replace each precision with the max precision to its right.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate over recall increments.
    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        if recall > previous_recall:
            ap += (recall - previous_recall) * precision
            previous_recall = recall
    return ap


@dataclass(frozen=True)
class Match:
    """One prediction's verdict: did it claim an unused ground truth?"""

    detection: Detection
    matched: bool
    best_iou: float


def match_detections(
    predictions: Iterable[Detection],
    ground_truth: Iterable[GroundTruth],
    iou_threshold: float = IOU_THRESHOLD,
) -> list[Match]:
    """Greedy, score-ordered, one-truth-per-prediction matching.

    This is the single matching rule. `mean_ap50` is built on it, and so is the
    overlay that colours boxes, so a picture of a frame and the number computed
    from it cannot disagree about which boxes were right.

    Returned grouped by label, each group in descending score order with a
    value-based tie-break, which is the order AP integrates over.
    """
    preds = list(predictions)
    truths_by_key: dict[tuple[str, int], list[GroundTruth]] = {}
    for truth in ground_truth:
        truths_by_key.setdefault((truth.image_id, truth.label), []).append(truth)

    matches: list[Match] = []
    for label in sorted({p.label for p in preds}):
        candidates = [p for p in preds if p.label == label]
        candidates.sort(key=lambda d: (-d.score, d.image_id, d.box))
        claimed: set[int] = set()
        for prediction in candidates:
            pool = truths_by_key.get((prediction.image_id, prediction.label), [])
            best_iou, best_index, best_any = 0.0, -1, 0.0
            for index, truth in enumerate(pool):
                overlap = iou_xyxy(prediction.box, truth.box)
                best_any = max(best_any, overlap)
                if id(truth) in claimed:
                    continue
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= iou_threshold:
                claimed.add(id(pool[best_index]))
                matches.append(Match(prediction, True, best_iou))
            else:
                matches.append(Match(prediction, False, best_any))
    return matches


def mean_ap50(
    predictions: Iterable[Detection],
    ground_truth: Iterable[GroundTruth],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, float]:
    """mAP@50 plus the counts behind it.

    Returning the counts alongside the score is deliberate: a boundary report
    that says only "mAP fell to 0.19" is much less actionable than one that can
    also say the pipeline stopped detecting rather than started hallucinating.
    """
    preds = list(predictions)
    truths = list(ground_truth)

    gt_count_by_label: dict[int, int] = {}
    for truth in truths:
        gt_count_by_label[truth.label] = gt_count_by_label.get(truth.label, 0) + 1

    matches = match_detections(preds, truths, iou_threshold)
    sequence_by_label: dict[int, list[int]] = {}
    for m in matches:
        sequence_by_label.setdefault(m.detection.label, []).append(1 if m.matched else 0)

    per_class: dict[int, float] = {}
    for label, n_gt in gt_count_by_label.items():
        per_class[label] = average_precision(sequence_by_label.get(label, []), n_gt)

    total_tp = sum(1 for m in matches if m.matched)
    total_fp = len(matches) - total_tp
    map50 = sum(per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "mAP50": map50,
        "n_ground_truth": len(truths),
        "n_predictions": len(preds),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": len(truths) - total_tp,
        "n_classes_scored": len(per_class),
    }
