"""Every pipeline under test honours the same contract (W2-2).

Model independence is only a claim until the same checks pass on detectors
that differ in family, head and input size. Each is skipped cleanly if its
model has not been fetched.
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import pytest

from blindspot.runner.registry import PIPELINES, load_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "val" / "road100" / "manifest.json"


def available():
    return [name for name, spec in sorted(PIPELINES.items()) if (ROOT / spec["model"]).is_file()]


def real_frame():
    if not MANIFEST.is_file():
        pytest.skip("road100 not fetched")
    record = json.loads(MANIFEST.read_text())["images"][0]
    path = MANIFEST.parent / "images" / record["file_name"]
    if not path.is_file():
        pytest.skip("images not downloaded")
    return record["image_id"], cv2.imread(str(path))


def test_three_pipelines_are_registered():
    assert len(PIPELINES) >= 3
    families = {spec["family"] for spec in PIPELINES.values()}
    assert len(families) >= 2, "model independence needs more than one detector family"


@pytest.mark.parametrize("name", available())
def test_describes_its_provenance(name):
    d = load_pipeline(name).describe()
    assert d["name"] == name and d["backend"] == "cv::dnn" and d["license"]


@pytest.mark.parametrize("name", available())
def test_blank_image_gives_no_detections(name):
    assert load_pipeline(name).predict(np.full((480, 640, 3), 128, np.uint8), "blank") == []


@pytest.mark.parametrize("name", available())
def test_detects_objects_inside_the_frame(name):
    image_id, image = real_frame()
    dets = load_pipeline(name).predict(image, image_id)
    assert dets, f"{name} found nothing in an annotated street frame"
    h, w = image.shape[:2]
    for d in dets:
        x0, y0, x1, y1 = d.box
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert 0.0 <= d.score <= 1.0 and 0 <= d.label < 80


@pytest.mark.parametrize("name", available())
def test_is_deterministic(name):
    image_id, image = real_frame()
    p = load_pipeline(name)
    assert p.predict(image, image_id) == p.predict(image, image_id)


@pytest.mark.parametrize("name", available())
def test_boxes_land_on_the_right_objects(name):
    """Catches decode and letterbox bugs: on an undegraded frame the pipeline
    must match a reasonable share of ground truth, not merely emit boxes."""
    from blindspot.metrics import mean_ap50
    from blindspot.runner.dataset import ValidationSet
    from blindspot.runner.yolox import COCO_CLASSES

    if not MANIFEST.is_file():
        pytest.skip("road100 not fetched")
    vs = ValidationSet(MANIFEST.parent, COCO_CLASSES)
    frames = vs.load(limit=10)
    p = load_pipeline(name)
    preds = [d for f in frames for d in vs.filter_predictions(p.predict(f.image, f.image_id))]
    truths = [t for f in frames for t in f.truths]
    assert mean_ap50(preds, truths)["mAP50"] > 0.2
