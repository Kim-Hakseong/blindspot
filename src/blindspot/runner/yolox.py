"""YOLOX-S under `cv::dnn`, used as a reference pipeline under test.

The model is the Apache-2.0 YOLOX-S export from the OpenCV model zoo. It is a
*subject* of measurement here, not a component of the tool: Blindspot's claims
are about the method, and the same measurement runs against any pipeline that
implements `Pipeline`.

Two details that decide whether the numbers mean anything:

- **Letterbox, not stretch.** The image is resized preserving aspect ratio and
  padded, then boxes are mapped back through the same transform. Stretching to
  640x640 would change object aspect ratios and would itself be a degradation,
  confounding every axis we are trying to measure.
- **Raw grid decode.** The export emits undecoded predictions over the 8/16/32
  stride grids: centres as grid offsets, sizes in log space, objectness and
  class scores already sigmoid-activated.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

from ..metrics import Detection
from .base import Pipeline

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
PAD_VALUE = 114  # YOLOX's standard letterbox fill.

#: COCO's 80 classes in the order the model emits them.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def _build_grid(size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Grid centres and strides for every prediction cell at this input size."""
    grids, expanded = [], []
    for stride in STRIDES:
        n = size // stride
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack([xx.ravel(), yy.ravel()], axis=1))
        expanded.append(np.full((n * n, 1), stride))
    return (
        np.concatenate(grids).astype(np.float32),
        np.concatenate(expanded).astype(np.float32),
    )


_GRID, _EXPANDED_STRIDES = _build_grid()


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Resize preserving aspect ratio and pad to a square. Returns (canvas, scale)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


class YoloxPipeline(Pipeline):
    name = "yolox_s"

    def __init__(
        self,
        model_path: str | pathlib.Path = "models/yolox_s.onnx",
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        classes: tuple[str, ...] = COCO_CLASSES,
        input_size: int = INPUT_SIZE,
        name: str = "yolox_s",
        license: str = "Apache-2.0 (OpenCV model zoo)",
    ) -> None:
        self.name = name
        self.license = license
        self.input_size = input_size
        if input_size == INPUT_SIZE:
            self._grid, self._strides = _GRID, _EXPANDED_STRIDES
        else:
            self._grid, self._strides = _build_grid(input_size)
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"{self.model_path} not found. Fetch it with:\n"
                "  curl -L -o models/yolox_s.onnx https://github.com/opencv/opencv_zoo/"
                "raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
            )
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def describe(self) -> dict[str, str]:
        return {
            "name": self.name,
            "model_path": str(self.model_path),
            "backend": "cv::dnn",
            "input_size": f"{self.input_size}x{self.input_size}",
            "license": self.license,
            "score_threshold": str(self.score_threshold),
            "nms_threshold": str(self.nms_threshold),
        }

    def predict(self, image: np.ndarray, image_id: str) -> list[Detection]:
        canvas, scale = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas)
        self.net.setInput(blob)
        raw = self.net.forward()[0]  # (8400, 85)

        # Decode: centres are grid offsets, sizes are log-scale, both in strides.
        centres = (raw[:, :2] + self._grid) * self._strides
        sizes = np.exp(raw[:, 2:4]) * self._strides

        objectness = raw[:, 4]
        class_scores = raw[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = objectness * class_scores[np.arange(len(class_ids)), class_ids]

        keep = confidences >= self.score_threshold
        if not np.any(keep):
            return []

        centres, sizes = centres[keep], sizes[keep]
        confidences, class_ids = confidences[keep], class_ids[keep]

        # Undo the letterbox to get coordinates in the original image frame.
        x0 = (centres[:, 0] - sizes[:, 0] / 2.0) / scale
        y0 = (centres[:, 1] - sizes[:, 1] / 2.0) / scale
        widths = sizes[:, 0] / scale
        heights = sizes[:, 1] / scale

        boxes = np.stack([x0, y0, widths, heights], axis=1)

        # Class-aware NMS: offset each class into its own coordinate band so
        # overlapping objects of different classes never suppress each other.
        offsets = class_ids.astype(np.float32) * 10000.0
        shifted = boxes.copy()
        shifted[:, 0] += offsets

        indices = cv2.dnn.NMSBoxes(
            shifted.tolist(),
            confidences.astype(np.float32).tolist(),
            self.score_threshold,
            self.nms_threshold,
        )
        if len(indices) == 0:
            return []

        h, w = image.shape[:2]
        detections = []
        for i in np.array(indices).ravel():
            bx, by, bw, bh = boxes[i]
            detections.append(
                Detection(
                    image_id=image_id,
                    label=int(class_ids[i]),
                    score=float(confidences[i]),
                    box=(
                        float(np.clip(bx, 0, w)),
                        float(np.clip(by, 0, h)),
                        float(np.clip(bx + bw, 0, w)),
                        float(np.clip(by + bh, 0, h)),
                    ),
                )
            )
        # Stable order so a probe's output does not depend on NMS tie-breaking.
        detections.sort(key=lambda d: (-d.score, d.label, d.box))
        return detections
