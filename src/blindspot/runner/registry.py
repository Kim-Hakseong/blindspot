"""The pipelines Blindspot has been measured against.

Each entry records where its weights come from and under what licence, so a
report can say exactly what was interrogated. Two share the YOLOX family but
differ tenfold in capacity; NanoDet is a different family and head.
"""

from __future__ import annotations

PIPELINES = {
    "yolox_s": {
        "model": "models/yolox_s.onnx", "family": "YOLOX", "input": 640,
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        "license": "Apache-2.0 (OpenCV model zoo)",
    },
    "yolox_nano": {
        "model": "models/yolox_nano.onnx", "family": "YOLOX", "input": 416,
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "license": "Apache-2.0 (Megvii YOLOX release)",
    },
    "nanodet_plus_m": {
        "model": "models/nanodet_plus_m.onnx", "family": "NanoDet", "input": 416,
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx",
        "license": "Apache-2.0 (OpenCV model zoo)",
    },
}


def load_pipeline(name: str):
    spec = PIPELINES[name]
    if spec["family"] == "NanoDet":
        from .nanodet import NanoDetPipeline

        return NanoDetPipeline(spec["model"])
    from .yolox import YoloxPipeline

    return YoloxPipeline(spec["model"], input_size=spec["input"], name=name, license=spec["license"])
