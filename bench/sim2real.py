"""Sim-to-real gap on real low-light photos from ExDark (W6-7).

    sh tools/fetch_exdark.sh                    # checksummed download, not committed
    uv run --group dev python bench/sim2real.py --out bench/out/sim2real.json

No first-party capture was made; see src/blindspot/sim2real.py for the method
and DATASETS.md for the licence. Illuminance is estimated, never measured.

Selection: ExDark images labelled Car or Bus, outdoor (ExDark's own
indoor/outdoor column), with at least one Car or Bus box, excluding any image
whose largest annotated object is a person (rule C5). Scored classes: ExDark
Car -> COCO car (YOLOX "truck" predictions also count as car, since ExDark has
no truck class and labels trucks as cars), Bus -> COCO bus. Images are read
and measured with OpenCV; Pillow reads only EXIF metadata (file I/O, rule T5).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.boundary.coverage import _invert, population_sweep  # noqa: E402
from blindspot.measure import snr_db  # noqa: E402
from blindspot.metrics import Detection, GroundTruth, mean_ap50  # noqa: E402
from blindspot.runner.dataset import ValidationSet  # noqa: E402
from blindspot.runner.registry import load_pipeline  # noqa: E402
from blindspot.runner.yolox import COCO_CLASSES  # noqa: E402
from blindspot.sim2real import (  # noqa: E402
    EXIF_METHOD, STATS_METHOD, compare_bins, exif_illuminance_lux, parse_exdark,
    person_is_main_subject,
)

EXDARK = ROOT / ".cache" / "exdark"
CAR, BUS, TRUCK = (COCO_CLASSES.index(c) for c in ("car", "bus", "truck"))
GT_LABEL = {"Car": CAR, "Bus": BUS}
MEASURE_MAX_SIDE = 640  # the synthetic sweep's frames are COCO-sized; noise is scale-dependent
SEED = 20260906


def exif(path: pathlib.Path) -> dict:
    from PIL import Image  # metadata only

    try:
        ex = Image.open(path).getexif().get_ifd(0x8769)
    except Exception:
        return {}
    t, n, iso = ex.get(0x829A), ex.get(0x829D), ex.get(0x8827)
    return {"exposure_s": float(t) if t else None, "f_number": float(n) if n else None,
            "iso": float(iso[0] if isinstance(iso, tuple) else iso) if iso else None}


def select(classlist: pathlib.Path) -> list[dict]:
    rows = [line.split() for line in classlist.read_text().splitlines()[1:] if line.strip()]
    chosen = []
    for name, cls, light, inout, _split in rows:
        if cls not in ("4", "5") or inout != "2":  # Bus(4), Car(5); Outdoor(2)
            continue
        folder = "Bus" if cls == "4" else "Car"
        ann = EXDARK / "ExDark_Annno" / folder / f"{name}.txt"
        img = next((p for p in (EXDARK / "ExDark" / folder).glob(f"{pathlib.Path(name).stem}.*")), None)
        if not ann.is_file() or img is None:
            continue
        boxes = parse_exdark(ann.read_text())
        if not any(b["cls"] in GT_LABEL for b in boxes) or person_is_main_subject(boxes):
            continue
        chosen.append({"name": name, "path": img, "boxes": boxes, "lighting_type": int(light)})
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "bench" / "out" / "sim2real.json")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    eff = json.loads((ROOT / "bench" / "out" / "efficiency.json").read_text())
    axis = next(a for a in eff["axes"] if a["axis"] == "low_light.illuminance_lux")
    threshold = eff["criterion"]["threshold_map50"]

    # SNR -> lux mapping: Blindspot's own population sweep on road100 (coverage method).
    road = ValidationSet(ROOT / "val" / "road100", COCO_CLASSES).load()
    values = np.array([p["value"] for p in axis["map50_curve"]], dtype=float)
    sweep = np.array(population_sweep(road, "low_light", "illuminance_lux", values, "snr_db", SEED))

    pipeline = load_pipeline("yolox_s")
    images = select(EXDARK / "imageclasslist.txt")
    records = []
    for im in images:
        frame = cv2.imread(str(im["path"]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = MEASURE_MAX_SIDE / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        e = exif(im["path"])
        preds = [Detection(d.image_id, CAR if d.label == TRUCK else d.label, d.score, d.box)
                 for d in pipeline.predict(frame, im["name"]) if d.label in (CAR, BUS, TRUCK)]
        truths = [GroundTruth(im["name"], GT_LABEL[b["cls"]], b["box"])
                  for b in im["boxes"] if b["cls"] in GT_LABEL]
        records.append({
            "name": im["name"], "lighting_type": im["lighting_type"],
            "lux_exif": exif_illuminance_lux(e.get("exposure_s"), e.get("f_number"), e.get("iso")),
            "lux_stats": _invert(values, sweep, snr_db(small)),
            "exif": e, "preds": preds, "truths": truths,
        })

    def by_estimate(key: str) -> dict:
        rs = sorted((r for r in records if r[key] is not None), key=lambda r: r[key])
        if len(rs) < args.bins:
            return {"images": len(rs), "note": "too few images with this estimate to bin"}
        bins = []
        for chunk in np.array_split(np.arange(len(rs)), args.bins):
            sel = [rs[i] for i in chunk]
            m = mean_ap50([p for r in sel for p in r["preds"]], [t for r in sel for t in r["truths"]])
            bins.append({"images": len(sel), "lux_min": sel[0][key], "lux_max": sel[-1][key],
                         "lux_median": float(np.median([r[key] for r in sel])),
                         "real_map50": m["mAP50"], "objects": sum(len(r["truths"]) for r in sel)})
        return {"images": len(rs), **compare_bins(bins, axis["map50_curve"], threshold)}

    pooled = mean_ap50([p for r in records for p in r["preds"]], [t for r in records for t in r["truths"]])
    report = {
        "first_party_capture": False,
        "dataset": {"name": "ExDark (Exclusively Dark)", "version": "repository terms v1.0, 2026-10-01",
                    "url": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
                    "licence": "annotations: non-commercial academic research/benchmarking terms; "
                               "images: third-party rights, not redistributed"},
        "pipeline": "yolox_s", "threshold_map50": threshold,
        "synthetic_boundary_lux": [axis["levels"][3]["grid"]["lower"], axis["levels"][3]["grid"]["upper"]],
        "selection": {"candidates": len(images), "measured": len(records),
                      "rule": "ExDark Car/Bus, outdoor, >=1 car/bus box, largest box not a person"},
        "objects": sum(len(r["truths"]) for r in records),
        "pooled_real_map50": pooled["mAP50"],
        "with_exif_estimate": sum(r["lux_exif"] is not None for r in records),
        "estimators": {
            "exif": {"method": EXIF_METHOD, **by_estimate("lux_exif")},
            "image_statistics": {"method": STATS_METHOD, **by_estimate("lux_stats")},
        },
        "command": "uv run --group dev python bench/sim2real.py",
    }
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "estimators"}, indent=1, default=str))
    for name, est in report["estimators"].items():
        print(name, est.get("images"), "agree", est.get("agree"), "disagree", est.get("disagree"))
        for b in est.get("bins", []):
            print(f"  {b['lux_min']:.2f}-{b['lux_max']:.2f} lux (est.) n={b['images']} "
                  f"real {b['real_map50']:.3f} synthetic {b['synthetic_map50']:.3f} "
                  f"{'FAIL' if b['real_failed'] else 'pass'}/{'FAIL' if b['synthetic_failed'] else 'pass'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
