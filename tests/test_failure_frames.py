"""Which frame gets shown (W2-7).

Picking the illustrative frame is where a demo can quietly cherry-pick, so the
rule is a pure function with its own tests: only redistributable frames, only
frames the pipeline got right when undegraded, and a stated fallback when no
frame shows a *confident* wrong box -- reported as such, not dressed up.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("failure_frames", ROOT / "bench" / "failure_frames.py")
failure_frames = importlib.util.module_from_spec(SPEC)
sys.modules["failure_frames"] = failure_frames
SPEC.loader.exec_module(failure_frames)

Stats = failure_frames.FrameStats
pick = failure_frames.pick_frame


def s(image_id, licence=4, base_tp=3, fail_tp=1, fail_fp_scores=()):
    # By default every wrong box is new, i.e. induced by the degradation.
    return Stats(image_id=image_id, license_id=licence, baseline_tp=base_tp,
                 fail_tp=fail_tp, fail_fp_scores=tuple(fail_fp_scores),
                 fail_new_fp_scores=tuple(fail_fp_scores))


def test_prefers_the_most_confident_wrong_box():
    chosen, mode = pick([s("a", fail_fp_scores=[0.55]), s("b", fail_fp_scores=[0.91])])
    assert chosen.image_id == "b"
    assert mode == "confident_false_positive"


def test_ignores_frames_that_cannot_be_redistributed():
    chosen, _ = pick([s("sa", licence=5, fail_fp_scores=[0.99]), s("by", fail_fp_scores=[0.6])])
    assert chosen.image_id == "by"


def test_ignores_frames_the_pipeline_already_missed_undegraded():
    chosen, _ = pick([s("never", base_tp=0, fail_fp_scores=[0.99]), s("ok", fail_fp_scores=[0.6])])
    assert chosen.image_id == "ok"


def test_low_confidence_false_positives_do_not_count_as_confident():
    chosen, mode = pick([s("a", base_tp=4, fail_tp=0, fail_fp_scores=[0.3])])
    assert mode == "missed_detections"
    assert chosen.image_id == "a"


def test_fallback_picks_the_largest_loss_of_true_positives():
    chosen, mode = pick([s("a", base_tp=3, fail_tp=2), s("b", base_tp=5, fail_tp=0)])
    assert mode == "missed_detections"
    assert chosen.image_id == "b"


def test_ties_break_deterministically_by_image_id():
    a, _ = pick([s("z", fail_fp_scores=[0.8]), s("m", fail_fp_scores=[0.8])])
    b, _ = pick([s("m", fail_fp_scores=[0.8]), s("z", fail_fp_scores=[0.8])])
    assert a.image_id == b.image_id == "m"


def test_returns_none_when_nothing_is_eligible():
    chosen, mode = pick([s("x", licence=1)])
    assert chosen is None and mode == "no_eligible_frame"


def test_a_wrong_box_already_present_undegraded_does_not_count():
    """Found on real data: frame 89556 has a 0.90 false positive with no
    degradation at all. Showing it as a degradation failure would blame the
    condition for something the detector does on a clean frame."""
    preexisting = s("old", fail_fp_scores=[0.95], base_tp=5, fail_tp=5)
    preexisting = Stats(**{**preexisting.__dict__, "fail_new_fp_scores": ()})
    induced = s("new", fail_fp_scores=[0.7])
    chosen, mode = pick([preexisting, induced])
    assert chosen.image_id == "new"
    assert mode == "confident_false_positive"


def test_only_preexisting_wrong_boxes_falls_back_to_missed_detections():
    only_old = Stats(image_id="old", license_id=4, baseline_tp=5, fail_tp=2,
                     fail_fp_scores=(0.95,), fail_new_fp_scores=())
    chosen, mode = pick([only_old])
    assert mode == "missed_detections"


def test_frames_where_people_are_the_subject_are_not_showcased():
    """Found on real data: the JPEG pick was a children's team photo. Blurring
    faces is not enough for a public showcase; a frame whose labelled objects
    are mostly people is not eligible to be shown at all."""
    portrait = Stats(image_id="team", license_id=4, baseline_tp=10, fail_tp=6,
                     fail_fp_scores=(0.95,), fail_new_fp_scores=(0.95,), person_share=0.9)
    street = Stats(image_id="street", license_id=4, baseline_tp=5, fail_tp=2,
                   fail_fp_scores=(0.6,), fail_new_fp_scores=(0.6,), person_share=0.3)
    chosen, _ = pick([portrait, street])
    assert chosen.image_id == "street"
