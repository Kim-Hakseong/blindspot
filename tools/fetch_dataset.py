"""Build a validation set from COCO val2017, filtered by image licence.

Blindspot creates *derivative* images: every probe is a degraded version of a
source frame. That rules out any source under a No-Derivatives licence, and a
competition entry rules out Non-Commercial. So this selects only images whose
COCO licence permits derivative works and commercial use:

    4  CC BY 2.0
    5  CC BY-SA 2.0
    7  No known copyright restrictions
    8  United States Government Work

Licences 1, 2 and 3 are Non-Commercial; 3 and 6 are No-Derivatives. All are
excluded. Selection is deterministic given the same annotations file and seed.

Pixels are never redistributed by this repository. The manifest records each
image's URL, licence, dimensions and SHA-256, and this script re-fetches them.

    uv run python tools/fetch_dataset.py --name road100 --count 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: COCO licence ids that permit derivative works and commercial use.
ALLOWED_LICENCES = {4, 5, 7, 8}

#: Category selections that make a coherent scene type.
PRESETS = {
    "road100": {
        "categories": ["car", "bus", "truck", "person", "traffic light", "motorcycle"],
        "description": "Street scenes with vehicles and pedestrians",
    },
    "indoor100": {
        "categories": ["person", "chair", "laptop", "tv", "bottle", "cup"],
        "description": "Indoor scenes with people and objects",
    },
    "retail100": {
        "categories": ["bottle", "cup", "bowl", "banana", "apple", "orange"],
        "description": "Shelf-like scenes with small repeated objects",
    },
}

MIN_OBJECTS = 3
MIN_BOX_AREA = 32 * 32  # Ignore objects too small to survive any degradation.


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def load_coco(path: pathlib.Path) -> dict:
    """Read COCO val2017 instances, downloading the official archive if absent."""
    if not path.is_file():
        import zipfile

        archive = path.parents[1] / "annotations_trainval2017.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            print(f"downloading COCO annotations (~240 MB) from {ANNOTATIONS_URL}")
            tmp = archive.with_suffix(".part")
            urllib.request.urlretrieve(ANNOTATIONS_URL, tmp)
            tmp.rename(archive)
        with zipfile.ZipFile(archive) as z:
            z.extract("annotations/instances_val2017.json", path.parents[1])
    return json.loads(path.read_text(encoding="utf-8"))


def select(coco: dict, preset: dict, count: int) -> tuple[list[dict], dict]:
    licences = {lic["id"]: lic for lic in coco["licenses"]}
    wanted_names = set(preset["categories"])
    wanted_ids = {c["id"] for c in coco["categories"] if c["name"] in wanted_names}
    category_names = {c["id"]: c["name"] for c in coco["categories"]}

    images = {img["id"]: img for img in coco["images"]}

    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        if ann["category_id"] not in wanted_ids:
            continue
        if ann["bbox"][2] * ann["bbox"][3] < MIN_BOX_AREA:
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)

    eligible = []
    for image_id, anns in by_image.items():
        image = images[image_id]
        if image["license"] not in ALLOWED_LICENCES:
            continue
        if len(anns) < MIN_OBJECTS:
            continue
        eligible.append((image_id, image, anns))

    # Deterministic ordering: most annotated first, ties broken by image id.
    eligible.sort(key=lambda item: (-len(item[2]), item[0]))
    chosen = eligible[:count]

    records = []
    for image_id, image, anns in chosen:
        records.append(
            {
                "image_id": str(image_id),
                "file_name": image["file_name"],
                "url": image["coco_url"].replace("https://", "http://"),
                # Original photograph, for the attribution CC BY requires when a
                # rendered frame is published (see DATASETS.md).
                "flickr_url": image.get("flickr_url"),
                "width": image["width"],
                "height": image["height"],
                "license_id": image["license"],
                "license_name": licences[image["license"]]["name"],
                "license_url": licences[image["license"]]["url"],
                "objects": [
                    {
                        "label": category_names[a["category_id"]],
                        "category_id": a["category_id"],
                        # COCO bbox is [x, y, w, h]; store xyxy in pixels.
                        "box": [
                            round(a["bbox"][0], 2),
                            round(a["bbox"][1], 2),
                            round(a["bbox"][0] + a["bbox"][2], 2),
                            round(a["bbox"][1] + a["bbox"][3], 2),
                        ],
                    }
                    for a in sorted(anns, key=lambda a: a["id"])
                ],
            }
        )

    stats = {
        "eligible_images": len(eligible),
        "selected": len(records),
        "total_objects": sum(len(r["objects"]) for r in records),
        "licence_breakdown": {},
    }
    for record in records:
        key = record["license_name"]
        stats["licence_breakdown"][key] = stats["licence_breakdown"].get(key, 0) + 1
    return records, stats


def download(records: list[dict], image_dir: pathlib.Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records, 1):
        target = image_dir / record["file_name"]
        if target.is_file():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(record["url"], timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
        record["bytes"] = len(data)
        if i % 20 == 0 or i == len(records):
            print(f"  {i}/{len(records)} images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="road100", choices=sorted(PRESETS))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--annotations",
        type=pathlib.Path,
        default=ROOT / ".cache" / "annotations" / "instances_val2017.json",
    )
    args = parser.parse_args()

    preset = PRESETS[args.name]
    print(f"reading {args.annotations.name}")
    coco = load_coco(args.annotations)

    records, stats = select(coco, preset, args.count)
    print(f"eligible under permitted licences: {stats['eligible_images']}")
    print(f"selected: {stats['selected']} images, {stats['total_objects']} objects")
    for licence, n in sorted(stats["licence_breakdown"].items()):
        print(f"  {n:>4}  {licence}")

    out_dir = ROOT / "val" / args.name
    print(f"downloading into {out_dir / 'images'}")
    download(records, out_dir / "images")

    manifest = {
        "name": args.name,
        "description": preset["description"],
        "source": "COCO val2017",
        "source_url": "https://cocodataset.org/",
        "annotations_license": "CC BY 4.0",
        "image_license_policy": (
            "Only COCO licence ids 4, 5, 7, 8 (CC BY 2.0, CC BY-SA 2.0, no known "
            "copyright restrictions, US Government Work). Non-Commercial and "
            "No-Derivatives licences are excluded because probes are derivative works."
        ),
        "redistributed": False,
        "categories": preset["categories"],
        "selection": {
            "min_objects": MIN_OBJECTS,
            "min_box_area_px": MIN_BOX_AREA,
            "order": "descending object count, ties by image id",
        },
        "stats": stats,
        "command": f"uv run python tools/fetch_dataset.py --name {args.name} --count {args.count}",
        "images": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
