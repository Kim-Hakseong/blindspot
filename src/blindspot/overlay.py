"""Evidence overlays: what the pipeline predicted, against what was there.

Predicted boxes are solid, coloured by whether they matched ground truth under
the same rule that produces mAP (`metrics.match_detections`). Ground truth is a
thin dashed outline. Each prediction carries its confidence, because the
failure this tool exists to expose is the *confident* wrong box -- a detector
does not say it has failed.

Colour is never the only signal: the verdict is also written in the label
("ok" / "miss"), so the overlay reads without colour vision.

Drawing is OpenCV only. The input image is never modified.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from .metrics import GroundTruth, Match

#: Design tokens, converted to OpenCV's BGR order.
COLORS_BGR = {
    "pass": (0x8C, 0xD6, 0x3D),   # --pass    #3DD68C
    "fail": (0x4D, 0x4D, 0xFF),   # --fail    #FF4D4D
    "truth": (0x84, 0x76, 0x6B),  # --text-2  #6B7684
    "text": (0xF2, 0xEC, 0xE8),   # --text-0  #E8ECF2
    "bar": (0x14, 0x0E, 0x0B),    # --bg-0    #0B0E14
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_HEIGHT = 32


def _dashed_rect(img, p0, p1, color, dash=6, gap=4, thickness=1):
    x0, y0 = p0
    x1, y1 = p1
    for (ax, ay), (bx, by) in (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ):
        length = int(np.hypot(bx - ax, by - ay))
        if length == 0:
            continue
        for start in range(0, length, dash + gap):
            end = min(start + dash, length)
            s = (int(ax + (bx - ax) * start / length), int(ay + (by - ay) * start / length))
            e = (int(ax + (bx - ax) * end / length), int(ay + (by - ay) * end / length))
            cv2.line(img, s, e, color, thickness, cv2.LINE_8)


def _label(img, text, origin, color):
    (w, h), base = cv2.getTextSize(text, FONT, 0.4, 1)
    x, y = origin
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y), COLORS_BGR["bar"], cv2.FILLED)
    cv2.putText(img, text, (x + 2, y - base), FONT, 0.4, color, 1, cv2.LINE_AA)


def draw_overlay(
    image: np.ndarray,
    matches: Sequence[Match],
    truths: Iterable[GroundTruth],
    caption: str,
    class_names: Sequence[str] | None = None,
    show_scores: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` with truths, predictions and a caption bar."""
    canvas = image.copy()

    for truth in truths:
        x0, y0, x1, y1 = (int(round(v)) for v in truth.box)
        _dashed_rect(canvas, (x0, y0), (x1, y1), COLORS_BGR["truth"])

    # Solid boxes first, labels after, so a label is never painted over.
    for m in matches:
        x0, y0, x1, y1 = (int(round(v)) for v in m.detection.box)
        color = COLORS_BGR["pass"] if m.matched else COLORS_BGR["fail"]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_8)

    if show_scores:
        for m in matches:
            x0, y0 = int(round(m.detection.box[0])), int(round(m.detection.box[1]))
            name = (
                class_names[m.detection.label]
                if class_names and m.detection.label < len(class_names)
                else str(m.detection.label)
            )
            verdict = "ok" if m.matched else "miss"
            color = COLORS_BGR["pass"] if m.matched else COLORS_BGR["fail"]
            _label(canvas, f"{name} {m.detection.score:.2f} {verdict}", (x0, y0), color)

    bar = np.zeros((CAPTION_HEIGHT, canvas.shape[1], 3), np.uint8)
    bar[:] = COLORS_BGR["bar"]
    if caption:
        cv2.putText(bar, caption, (8, 21), FONT, 0.5, COLORS_BGR["text"], 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])
