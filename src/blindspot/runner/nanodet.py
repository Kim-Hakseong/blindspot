"""NanoDet-Plus-m under `cv::dnn` -- a second detector family.

Anchor-free with a Generalized Focal Loss head: each cell predicts a discrete
distribution over distances to the four box sides, decoded as the expectation
over `reg_max + 1` bins. That is a genuinely different head from YOLOX's direct
regression, which is the point of measuring it.

Decode follows the OpenCV model zoo reference (`nanodet.py`) with one
correction. The reference splits outputs as interleaved `cls, box, cls, box`,
but under OpenCV 5 the network returns them ordered `cls, cls, cls, box, box,
box`, so that split would silently pair class maps with the wrong pyramid
level. Outputs are therefore selected by shape: 80 channels are class scores,
32 are box distributions, and larger maps are finer strides. The export also
has three levels (strides 8/16/32), not the four the reference lists.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

from ..metrics import Detection
from .base import Pipeline
from .yolox import COCO_CLASSES, letterbox

INPUT_SIZE = 416
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)


class NanoDetPipeline(Pipeline):
    name = "nanodet_plus_m"

    def __init__(self, model_path: str | pathlib.Path = "models/nanodet_plus_m.onnx",
                 score_threshold: float = 0.25, nms_threshold: float = 0.45) -> None:
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"{self.model_path} not found. Fetch it with:\n"
                "  curl -L -o models/nanodet_plus_m.onnx https://github.com/opencv/opencv_zoo/"
                "raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
            )
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(self.model_path))
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.project = np.arange(REG_MAX + 1, dtype=np.float32)

    def describe(self) -> dict[str, str]:
        return {
            "name": self.name, "model_path": str(self.model_path), "backend": "cv::dnn",
            "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}", "family": "NanoDet (GFL head)",
            "license": "Apache-2.0 (OpenCV model zoo)",
            "score_threshold": str(self.score_threshold), "nms_threshold": str(self.nms_threshold),
        }

    def predict(self, image: np.ndarray, image_id: str) -> list[Detection]:
        canvas, scale = letterbox(image, INPUT_SIZE)
        x = (canvas.astype(np.float32) - MEAN) / STD
        self.net.setInput(cv2.dnn.blobFromImage(x))
        outs = [o.reshape(o.shape[-2], o.shape[-1]) for o in self.net.forward(self.out_names)]

        cls_maps = sorted([o for o in outs if o.shape[1] == len(COCO_CLASSES)], key=lambda o: -o.shape[0])
        box_maps = sorted([o for o in outs if o.shape[1] == 4 * (REG_MAX + 1)], key=lambda o: -o.shape[0])

        boxes, scores = [], []
        for cls, reg in zip(cls_maps, box_maps):
            side = int(round(np.sqrt(cls.shape[0])))
            stride = INPUT_SIZE // side
            ys, xs = np.divmod(np.arange(side * side), side)
            cx = xs * stride + 0.5 * (stride - 1)
            cy = ys * stride + 0.5 * (stride - 1)
            logits = reg.reshape(-1, REG_MAX + 1)
            prob = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            dist = (prob @ self.project).reshape(-1, 4) * stride
            boxes.append(np.stack([cx - dist[:, 0], cy - dist[:, 1],
                                   cx + dist[:, 2], cy + dist[:, 3]], axis=1))
            scores.append(cls)
        boxes = np.clip(np.concatenate(boxes), 0, INPUT_SIZE) / scale
        scores = np.concatenate(scores)
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        keep = confidences >= self.score_threshold
        if not np.any(keep):
            return []
        boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]

        xywh = np.column_stack([boxes[:, 0] + class_ids * 10000.0, boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), confidences.astype(np.float32).tolist(),
                               self.score_threshold, self.nms_threshold)
        h, w = image.shape[:2]
        out = []
        for i in np.array(idx).ravel():
            x0, y0, x1, y1 = boxes[i]
            x0, x1 = float(np.clip(x0, 0, w)), float(np.clip(x1, 0, w))
            y0, y1 = float(np.clip(y0, 0, h)), float(np.clip(y1, 0, h))
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(Detection(image_id=image_id, label=int(class_ids[i]),
                                 score=float(confidences[i]), box=(x0, y0, x1, y1)))
        out.sort(key=lambda d: (-d.score, d.label, d.box))
        return out
