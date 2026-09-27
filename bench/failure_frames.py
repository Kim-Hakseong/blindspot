"""Render one evidence frame per axis: undegraded vs. just past the boundary.

The condition shown is the first failing value the grid measured on each axis
(`bench/out/efficiency.json`, finest level), so the picture sits exactly where
the number says the pipeline breaks.

Which *frame* is shown is decided by `pick_frame`, a stated rule rather than a
judgement call:

1. only CC BY 2.0 frames (redistributable with attribution),
2. only frames the pipeline got at least partly right undegraded, and only
   frames where people are not the subject (under half the labelled objects
   are persons) -- blurred faces are not enough to put a group portrait, let
   alone one of children, in a public showcase,
3. prefer the frame whose most confident *degradation-induced* wrong box is
   most confident -- the silent failure this tool is about. A wrong box only
   counts if no prediction of the same class sat there undegraded (IoU < 0.5):
   on real data some frames carry a 0.90 false positive with no degradation at
   all, and showing those would blame the condition for the detector's
   ordinary behaviour,
4. if no frame has an induced wrong box at >= 0.5 confidence, fall back to the
   largest loss of correct detections, and say so in the output.

The frame is illustrative. The measured claim is the set-level mAP@50 in the
efficiency report; the JSON written here says that too.

Faces are blurred at render time only, after scoring.

    uv run python bench/failure_frames.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from blindspot.degrade import REGISTRY  # noqa: E402
from blindspot.metrics import iou_xyxy, match_detections  # noqa: E402
from blindspot.overlay import draw_overlay  # noqa: E402

CONFIDENT = 0.5
#: Boxes below this are not drawn, on either side, and the caption says so.
#: Matching and every count are computed on all predictions regardless.
SHOWN = 0.5
REDISTRIBUTABLE_LICENCES = {4}  # CC BY 2.0
MAX_PERSON_SHARE = 0.5


@dataclass(frozen=True)
class FrameStats:
    image_id: str
    license_id: int
    baseline_tp: int
    fail_tp: int
    fail_fp_scores: tuple[float, ...]
    #: Wrong boxes with no same-class undegraded prediction at IoU >= 0.5.
    fail_new_fp_scores: tuple[float, ...] = ()
    #: Fraction of the frame's labelled objects that are people.
    person_share: float = 0.0

    @property
    def max_fp(self) -> float:
        return max(self.fail_new_fp_scores, default=0.0)


def pick_frame(stats: list[FrameStats]) -> tuple[FrameStats | None, str]:
    eligible = [
        s for s in stats
        if s.license_id in REDISTRIBUTABLE_LICENCES
        and s.baseline_tp > 0
        and s.person_share < MAX_PERSON_SHARE
    ]
    if not eligible:
        return None, "no_eligible_frame"

    confident = [s for s in eligible if s.max_fp >= CONFIDENT]
    if confident:
        confident.sort(key=lambda s: (-s.max_fp, s.image_id))
        return confident[0], "confident_false_positive"

    eligible.sort(key=lambda s: (-(s.baseline_tp - s.fail_tp), s.image_id))
    return eligible[0], "missed_detections"


def _fail_value(entry: dict, axis) -> float:
    level = max(entry["levels"], key=lambda l: l["grid_steps"])
    grid = level["grid"]
    return grid["upper"] if axis.severe_end == "hi" else grid["lower"]


def main() -> int:
    from blindspot.privacy import FaceBlurrer
    from blindspot.runner.dataset import ValidationSet
    from blindspot.runner.yolox import COCO_CLASSES, YoloxPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficiency", type=pathlib.Path, default=pathlib.Path("bench/out/efficiency.json"))
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("val/road100"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/frames"))
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    efficiency = json.loads(args.efficiency.read_text(encoding="utf-8"))
    vs = ValidationSet(args.dataset, COCO_CLASSES)
    frames = vs.load()
    records = {r["image_id"]: r for r in vs.manifest["images"]}
    pipeline = YoloxPipeline()
    blur = FaceBlurrer()
    args.out.mkdir(parents=True, exist_ok=True)

    baseline = {}
    for f in frames:
        preds = vs.filter_predictions(pipeline.predict(f.image, f.image_id))
        baseline[f.image_id] = (preds, match_detections(preds, f.truths))

    index = []
    for entry in efficiency["axes"]:
        degradation, _, field = entry["axis"].partition(".")
        deg = REGISTRY.get(degradation)
        axis = deg.axis(field)
        value = _fail_value(entry, axis)
        values = {a.field: (a.lo + a.hi) / 2.0 for a in deg.axes}
        values[field] = value
        params = deg.Params(**values)

        stats, degraded = [], {}
        for f in frames:
            img = deg.apply(f.image, params, seed=args.seed)
            preds = vs.filter_predictions(pipeline.predict(img, f.image_id))
            m = match_detections(preds, f.truths)
            degraded[f.image_id] = (img, m)
            stats.append(FrameStats(
                image_id=f.image_id,
                license_id=records[f.image_id]["license_id"],
                baseline_tp=sum(x.matched for x in baseline[f.image_id][1]),
                person_share=(
                    sum(o["label"] == "person" for o in records[f.image_id]["objects"])
                    / max(len(records[f.image_id]["objects"]), 1)
                ),
                fail_tp=sum(x.matched for x in m),
                fail_fp_scores=tuple(round(x.detection.score, 4) for x in m if not x.matched),
                fail_new_fp_scores=tuple(
                    round(x.detection.score, 4)
                    for x in m
                    if not x.matched and not any(
                        b.label == x.detection.label and iou_xyxy(b.box, x.detection.box) >= 0.5
                        for b in baseline[f.image_id][0]
                    )
                ),
            ))

        chosen, mode = pick_frame(stats)
        if chosen is None:
            print(f"{entry['axis']}: no eligible frame")
            continue

        frame = next(f for f in frames if f.image_id == chosen.image_id)
        derived = " ".join(f"{k}={v:.3g}{u}" for k, (v, u) in deg.derived(params).items()
                           if u != "dimensionless")
        shown = lambda ms: [x for x in ms if x.detection.score >= SHOWN]  # noqa: E731
        left = draw_overlay(blur(frame.image), shown(baseline[chosen.image_id][1]), frame.truths,
                            caption=f"undegraded   (boxes >= {SHOWN:.2f} shown)",
                            class_names=COCO_CLASSES)
        img, m = degraded[chosen.image_id]
        right = draw_overlay(blur(img), shown(m), frame.truths,
                             caption=f"{field} = {value:.4g} {axis.unit}  {derived}".strip(),
                             class_names=COCO_CLASSES)
        sep = np.zeros((left.shape[0], 4, 3), np.uint8)
        sep[:] = (0x4D, 0x4D, 0xFF)
        png = args.out / f"{degradation}_{field}.png"
        cv2.imwrite(str(png), np.hstack([left, sep, right]))

        record = records[chosen.image_id]
        index.append({
            "axis": entry["axis"],
            "unit": axis.unit,
            "condition_value": value,
            "png": png.name,
            "selection_mode": mode,
            "selection_rule": "see bench/failure_frames.py::pick_frame",
            "frame": asdict(chosen) | {"max_fp_score": chosen.max_fp},
            "illustrative_only": True,
            "display_min_score": SHOWN,
            "measured_claim": "set-level mAP@50 in bench/out/efficiency.json",
            "attribution": {
                "source": record["flickr_url"],
                "license": record["license_name"],
                "license_url": record["license_url"],
                "modified": "resized-free overlay; faces blurred; degradation applied on right",
            },
            "reproduce": (
                f"uv run blindspot probe --degradation {degradation} --axis {field} "
                f"--value {float(value)!r} --seed {args.seed}"
            ),
        })
        print(f"{entry['axis']}: {chosen.image_id} ({mode}, max induced FP conf "
              f"{chosen.max_fp:.2f}, TP {chosen.baseline_tp}->{chosen.fail_tp}) -> {png}")

    (args.out / "frames.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
