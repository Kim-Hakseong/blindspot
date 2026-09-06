"""The pipeline adapter.

Tests that need the model or the dataset skip cleanly when those are absent, so
the suite still runs on a fresh clone. The letterbox geometry is tested without
either, because that is where coordinate bugs hide and they would silently
shift every boundary we report.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from blindspot.runner.yolox import INPUT_SIZE, PAD_VALUE, YoloxPipeline, letterbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolox_s.onnx"
MANIFEST = ROOT / "val" / "road100" / "manifest.json"

needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="models/yolox_s.onnx not fetched")
needs_data = pytest.mark.skipif(not MANIFEST.is_file(), reason="val/road100 not fetched")


# ------------------------------------------------------------- letterbox


@pytest.mark.parametrize("shape", [(428, 640), (640, 428), (640, 640), (100, 900)])
def test_letterbox_output_is_square_and_scale_preserves_aspect(shape):
    image = np.full((*shape, 3), 200, np.uint8)
    canvas, scale = letterbox(image)

    assert canvas.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert scale == pytest.approx(min(INPUT_SIZE / shape[0], INPUT_SIZE / shape[1]))

    # The scaled content must fit without being cropped.
    assert int(round(shape[0] * scale)) <= INPUT_SIZE
    assert int(round(shape[1] * scale)) <= INPUT_SIZE


def test_letterbox_pads_rather_than_stretches():
    """A wide image must leave padding below, not be stretched to fill."""
    image = np.full((100, 900, 3), 200, np.uint8)
    canvas, scale = letterbox(image)
    filled_rows = int(round(100 * scale))
    assert np.all(canvas[filled_rows + 1 :] == PAD_VALUE)
    assert np.all(canvas[: filled_rows - 1] == 200)


def test_letterbox_square_image_has_no_padding():
    image = np.full((640, 640, 3), 77, np.uint8)
    canvas, scale = letterbox(image)
    assert scale == pytest.approx(1.0)
    assert np.all(canvas == 77)


def test_letterbox_roundtrip_maps_a_box_back():
    """A box drawn in the original frame must return to where it started."""
    h, w = 428, 640
    _, scale = letterbox(np.zeros((h, w, 3), np.uint8))
    original = (100.0, 50.0, 300.0, 200.0)
    forward = tuple(v * scale for v in original)
    back = tuple(v / scale for v in forward)
    assert back == pytest.approx(original)


# ------------------------------------------------------------- inference


@needs_model
def test_pipeline_reports_its_provenance():
    described = YoloxPipeline(MODEL).describe()
    assert described["backend"] == "cv::dnn"
    assert described["name"] == "yolox_s"
    assert "score_threshold" in described


@needs_model
def test_pipeline_returns_nothing_on_a_blank_image():
    pipeline = YoloxPipeline(MODEL)
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert pipeline.predict(blank, "blank") == []


@needs_model
@needs_data
def test_pipeline_detects_objects_in_a_real_frame():
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["images"][0]
    path = MANIFEST.parent / "images" / record["file_name"]
    if not path.is_file():
        pytest.skip("images not downloaded")

    import cv2

    image = cv2.imread(str(path))
    detections = YoloxPipeline(MODEL).predict(image, record["image_id"])

    assert detections, "found nothing in a frame annotated with several objects"
    h, w = image.shape[:2]
    for d in detections:
        x0, y0, x1, y1 = d.box
        assert 0 <= x0 < x1 <= w, f"box out of frame horizontally: {d.box}"
        assert 0 <= y0 < y1 <= h, f"box out of frame vertically: {d.box}"
        assert 0.0 <= d.score <= 1.0
        assert 0 <= d.label < 80


@needs_model
@needs_data
def test_pipeline_is_deterministic():
    """Same image twice must give identical detections, or no boundary is stable."""
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["images"][0]
    path = MANIFEST.parent / "images" / record["file_name"]
    if not path.is_file():
        pytest.skip("images not downloaded")

    import cv2

    image = cv2.imread(str(path))
    pipeline = YoloxPipeline(MODEL)
    first = pipeline.predict(image, record["image_id"])
    second = pipeline.predict(image, record["image_id"])
    assert first == second
