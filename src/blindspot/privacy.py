"""Face anonymisation for anything that leaves the machine as a picture.

Validation frames contain people. Scoring always runs on the untouched frame;
blurring happens only when a frame is rendered for a report, a demo or a
video, so anonymisation can never move a measured number.

Faces are found with YuNet through OpenCV's own `FaceDetectorYN` (model from
the OpenCV model zoo). If the model is not present the caller is told, rather
than silently shipping unblurred frames.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

Box = tuple[float, float, float, float]


def blur_regions(image: np.ndarray, boxes: Iterable[Box], margin: float = 0.25) -> np.ndarray:
    """Return a copy with each (x0, y0, x1, y1) region heavily blurred."""
    out = image.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        ax0 = int(max(0, np.floor(x0 - bw * margin)))
        ay0 = int(max(0, np.floor(y0 - bh * margin)))
        ax1 = int(min(w, np.ceil(x1 + bw * margin)))
        ay1 = int(min(h, np.ceil(y1 + bh * margin)))
        if ax1 - ax0 < 2 or ay1 - ay0 < 2:
            continue
        region = out[ay0:ay1, ax0:ax1]
        k = max(3, (max(ax1 - ax0, ay1 - ay0) // 3) | 1)
        out[ay0:ay1, ax0:ax1] = cv2.GaussianBlur(region, (k, k), 0)
    return out


class FaceBlurrer:
    """Detect faces with YuNet and blur them."""

    def __init__(self, model_path: str | pathlib.Path = "models/face_detection_yunet.onnx",
                 score_threshold: float = 0.6) -> None:
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"{self.model_path} not found; refusing to render unanonymised frames.\n"
                f"  curl -L -o {self.model_path} {YUNET_URL}"
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def faces(self, image: np.ndarray) -> list[Box]:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, found = self.detector.detect(image)
        if found is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])) for f in found]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return blur_regions(image, self.faces(image))
