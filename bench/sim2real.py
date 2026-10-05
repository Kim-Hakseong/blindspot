"""Sim-to-real gap on real low-light photos from a third-party dataset (W6-7).

    uv run python tools/fetch_nod.py --drive-file-ids <ids>   # selected images only, not committed
    uv run --group dev python bench/sim2real.py --dataset nod   # -> bench/out/sim2real.json

No first-party capture was made; see src/blindspot/sim2real.py for the method.
Illuminance is estimated, never measured. Exposure time, aperture and ISO are
the values the camera recorded in EXIF.

Datasets:
- nod (published): NOD, Morawski et al., BMVC 2021,
  https://github.com/igor-morawski/NOD, images CC BY-NC-SA 2.0 as declared in
  its annotation files; used as a non-commercial research benchmark, aggregate
  metrics only, no image or derived image is ever published.
- exdark (not published): ExDark; its numbers stay in .cache/ until the
  project owner decides on its terms' §3.

Scoring: the dataset's vehicle class against YOLOX-S predictions of car,
truck and bus (NOD and ExDark label vehicles as "car"/"Car"; a sensitivity
run counting only the "car" prediction is reported alongside). Images whose
largest annotated object is a person are excluded (rule C5).

Synthetic prediction for each bin of EXIF-estimated illuminance: Blindspot's
low-light degradation applied to road100 at the bin's median estimated lux
*and* the median exposure time the cameras recorded, scored the same way the
boundary was. ISO is not mapped: the model's gain is in uncalibrated units.
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
sys.path.insert(0, str(ROOT / "tools"))

from blindspot.boundary.coverage import _invert, population_sweep  # noqa: E402
from blindspot.boundary.probe import run_condition_probe  # noqa: E402
from blindspot.degrade.compose import Condition  # noqa: E402
from blindspot.measure import snr_db  # noqa: E402
from blindspot.metrics import Detection, GroundTruth, mean_ap50  # noqa: E402
from blindspot.runner.dataset import ValidationSet  # noqa: E402
from blindspot.runner.registry import load_pipeline  # noqa: E402
from blindspot.runner.yolox import COCO_CLASSES  # noqa: E402
from blindspot.sim2real import (  # noqa: E402
    EXIF_CORRECTED_METHOD, EXIF_METHOD, STATS_METHOD, compare_bins, gap_summary, exif_illuminance_lux, parse_exdark,
    person_is_main_subject,
)

CAR, BUS, TRUCK = (COCO_CLASSES.index(c) for c in ("car", "bus", "truck"))
MEASURE_MAX_SIDE = 640  # the synthetic sweep's frames are COCO-sized; noise is scale-dependent
SEED = 20260906
LUX_RANGE = (0.5, 400.0)        # the synthetic low-light axis
EXPOSURE_MS_MAX = 1000.0        # LowLightParams.exposure_ms upper bound

NOD_CREDIT = {
    "name": "NOD (Night Object Detection) dataset",
    "citation": "I. Morawski, Y.-A. Chen, Y.-S. Lin, W. H. Hsu, \"NOD: Taking a Closer Look at "
                "Detection under Extreme Low-Light Conditions with Night Object Detection "
                "Dataset\", BMVC 2021, arXiv:2110.10364",
    "repository": "https://github.com/igor-morawski/NOD",
    "licence": "CC BY-NC-SA 2.0 (as declared in the NOD annotation files)",
    "use": "non-commercial research benchmark; aggregate metrics only; no NOD image or "
           "derived image is published",
}


def exif(path: pathlib.Path) -> dict:
    from PIL import Image  # metadata only (rule T5: file I/O convenience)

    try:
        ex = Image.open(path).getexif().get_ifd(0x8769)
    except Exception:
        return {}
    t, n, iso = ex.get(0x829A), ex.get(0x829D), ex.get(0x8827)
    return {"exposure_s": float(t) if t else None, "f_number": float(n) if n else None,
            "iso": float(iso[0] if isinstance(iso, tuple) else iso) if iso else None}


def load_nod() -> list[dict]:
    from fetch_nod import CACHE, selection

    out = []
    for name, info in selection().items():
        path = CACHE / "images" / name
        if not path.is_file():
            continue
        boxes = [{"cls": b["cls"], "box": (b["xywh"][0], b["xywh"][1],
                                           b["xywh"][0] + b["xywh"][2], b["xywh"][1] + b["xywh"][3])}
                 for b in info["boxes"]]
        out.append({"name": name, "path": path, "boxes": boxes})
    return out


def load_exdark() -> list[dict]:
    base = ROOT / ".cache" / "exdark"
    rows = [line.split() for line in (base / "imageclasslist.txt").read_text().splitlines()[1:] if line.strip()]
    out = []
    for name, cls, _light, inout, _split in rows:
        if cls not in ("4", "5") or inout != "2":  # Bus(4), Car(5); Outdoor(2)
            continue
        folder = "Bus" if cls == "4" else "Car"
        ann = base / "ExDark_Annno" / folder / f"{name}.txt"
        img = next(iter((base / "ExDark" / folder).glob(f"{pathlib.Path(name).stem}.*")), None)
        if not ann.is_file() or img is None:
            continue
        parsed = parse_exdark(ann.read_text())
        if person_is_main_subject(parsed):
            continue  # rule C5
        boxes = [b | {"cls": "car" if b["cls"] in ("Car", "Bus") else b["cls"]} for b in parsed]
        if any(b["cls"] == "car" for b in boxes):
            out.append({"name": name, "path": img, "boxes": boxes})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nod", "exdark"], default="nod")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()
    out_path = (ROOT / "bench" / "out" / "sim2real.json" if args.dataset == "nod"
                else ROOT / ".cache" / "exdark" / "sim2real_exdark.json")

    eff = json.loads((ROOT / "bench" / "out" / "efficiency.json").read_text())
    axis = next(a for a in eff["axes"] if a["axis"] == "low_light.illuminance_lux")
    threshold = eff["criterion"]["threshold_map50"]

    road_vs = ValidationSet(ROOT / "val" / "road100", COCO_CLASSES)
    road = road_vs.load()
    values = np.array([p["value"] for p in axis["map50_curve"]], dtype=float)
    sweep = np.array(population_sweep(road, "low_light", "illuminance_lux", values, "snr_db", SEED))
    pipeline = load_pipeline("yolox_s")

    images = load_nod() if args.dataset == "nod" else load_exdark()
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
        # Mean linear luminance (sRGB decoded with gamma 2.2), for the corrected EXIF estimate.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean_linear = float(np.mean(gray ** 2.2))
        raw = [d for d in pipeline.predict(frame, im["name"]) if d.label in (CAR, BUS, TRUCK)]
        truths = [GroundTruth(im["name"], CAR, b["box"]) for b in im["boxes"] if b["cls"] in ("car", "Car")]
        records.append({
            "name": im["name"],
            "lux_exif": exif_illuminance_lux(e.get("exposure_s"), e.get("f_number"), e.get("iso")),
            "lux_exif_corrected": exif_illuminance_lux(e.get("exposure_s"), e.get("f_number"), e.get("iso"),
                                                       mean_linear=mean_linear),
            "mean_linear_luminance": mean_linear,
            "lux_stats": _invert(values, sweep, snr_db(small)),
            "exif": e, "truths": truths,
            "preds_vehicle": [Detection(d.image_id, CAR, d.score, d.box) for d in raw],
            "preds_car_only": [d for d in raw if d.label == CAR],
        })

    def bin_records(key: str) -> list[list[dict]]:
        rs = sorted((r for r in records if r[key] is not None), key=lambda r: r[key])
        return [[rs[i] for i in chunk] for chunk in np.array_split(np.arange(len(rs)), args.bins)] \
            if len(rs) >= args.bins else []

    def score(sel, preds_key):
        return mean_ap50([p for r in sel for p in r[preds_key]], [t for r in sel for t in r["truths"]])["mAP50"]

    def summarise(key: str, matched: bool) -> dict:
        bins = []
        for sel in bin_records(key):
            lux = float(np.median([r[key] for r in sel]))
            b = {"images": len(sel), "objects": sum(len(r["truths"]) for r in sel),
                 "lux_min": sel[0][key], "lux_max": sel[-1][key], "lux_median": lux,
                 "real_map50": score(sel, "preds_vehicle"),
                 "real_map50_car_prediction_only": score(sel, "preds_car_only"),
                 "outside_synthetic_axis": not (LUX_RANGE[0] <= lux <= LUX_RANGE[1])}
            if matched:
                t_ms = float(np.median([r["exif"]["exposure_s"] for r in sel])) * 1000.0
                lux_c = float(np.clip(lux, *LUX_RANGE))
                t_c = min(t_ms, EXPOSURE_MS_MAX)
                cond = Condition.from_axes({"low_light.illuminance_lux": lux_c},
                                           fixed={"low_light.exposure_ms": t_c})
                b |= {"exposure_ms_median_recorded": t_ms, "synthetic_condition":
                      {"illuminance_lux": lux_c, "exposure_ms": t_c},
                      "synthetic_map50": run_condition_probe(road, pipeline, road_vs, cond, SEED)["map50"]}
            bins.append(b)
        n = sum(r[key] is not None for r in records)
        return {"images": n, **(compare_bins(bins, axis["map50_curve"], threshold) if bins else
                                {"note": "too few images with this estimate to bin"})}

    report = {
        "first_party_capture": False,
        "dataset": NOD_CREDIT if args.dataset == "nod" else {"name": "ExDark", "published": False},
        "pipeline": "yolox_s", "threshold_map50": threshold,
        "threshold_rule": "mAP@50 below 60% of the road100 baseline (the same criterion as the boundary)",
        "synthetic_boundary_lux": [axis["levels"][3]["grid"]["lower"], axis["levels"][3]["grid"]["upper"]],
        "synthetic_boundary_note": "located at the model's default exposure time, 10 ms",
        "selection": {"candidates": len(images), "measured": len(records),
                      "rule": "vehicle-labelled images; largest annotated object not a person"},
        "objects": sum(len(r["truths"]) for r in records),
        "pooled_real_map50": score(records, "preds_vehicle"),
        "pooled_real_map50_car_prediction_only": score(records, "preds_car_only"),
        "images_with_exif_exposure": sum(r["lux_exif"] is not None for r in records),
        "recorded_exposure_ms": {
            "median": float(np.median([r["exif"]["exposure_s"] * 1000 for r in records if r["exif"].get("exposure_s")]))
            if any(r["exif"].get("exposure_s") for r in records) else None,
            "label": "camera-recorded EXIF value (a setting, not an estimate)"},
        "primary_estimator": "exif_brightness_corrected",
        "estimators": {
            "exif_brightness_corrected": {
                "method": EXIF_CORRECTED_METHOD,
                "synthetic": "matched: bin's estimated lux and recorded exposure",
                **summarise("lux_exif_corrected", matched=True)},
            "exif": {"method": EXIF_METHOD, "synthetic": "matched: bin's estimated lux and recorded exposure",
                     **summarise("lux_exif", matched=True)},
            "image_statistics": {"method": STATS_METHOD, "synthetic": "default-exposure curve",
                                 **summarise("lux_stats", matched=False)},
        },
        "command": f"uv run --group dev python bench/sim2real.py --dataset {args.dataset}",
    }
    primary = report["estimators"][report["primary_estimator"]]
    if primary.get("bins"):
        report["gap"] = {"estimator": report["primary_estimator"],
                         **gap_summary(primary, tuple(report["synthetic_boundary_lux"]))}
    out_path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("estimators", "dataset")}, indent=1, default=str))
    for name, est in report["estimators"].items():
        print(name, est.get("images"), "agree", est.get("agree"), "disagree", est.get("disagree"))
        for b in est.get("bins", []):
            print(f"  {b['lux_min']:.3g}-{b['lux_max']:.3g} lux (est.) n={b['images']} real {b['real_map50']:.3f} "
                  f"synthetic {b['synthetic_map50']:.3f} curve {b['curve_map50']:.3f} "
                  f"{'FAIL' if b['real_failed'] else 'pass'}/{'FAIL' if b['synthetic_failed'] else 'pass'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
