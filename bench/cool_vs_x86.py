"""Same probe batch, different OpenCV build or CPU (rule K2).

Runs one fixed batch (`bench/fixtures/probe_batch_64.json`) and times each
stage separately -- degradation, objective measurement, `cv::dnn` inference --
so a speed-up can be attributed to where it happens rather than asserted for
the whole pipeline. It also fingerprints which OpenCV actually loaded (module
path, version, a hash of `cv2.getBuildInformation()`, whether KleidiCV is
compiled in), because "we ran COOL" is only a claim until the build info says
so.

Planned arms (see LOG W5-1): COOL AMI on c8g; stock OpenCV 5 wheel on the same
c8g; stock wheel on c7i. The first pair isolates COOL, the second the CPU.
Correctness is checked too: every arm must produce the same mAP per probe, or
the faster arm is computing something else.

    uv run python bench/cool_vs_x86.py --label local-m4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.degrade.compose import Condition, apply_condition  # noqa: E402
from blindspot.measure import measure  # noqa: E402
from blindspot.metrics import mean_ap50  # noqa: E402

FIXTURE = ROOT / "bench" / "fixtures" / "probe_batch_64.json"
AXES = ("motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m", "jpeg.quality")


def make_fixture(path: pathlib.Path) -> None:
    """64 single-axis conditions, 16 per axis across its range. Deterministic."""
    from blindspot.degrade import REGISTRY

    probes = []
    for axis_id in AXES:
        deg, _, field = axis_id.partition(".")
        a = REGISTRY.get(deg).axis(field)
        for k, v in enumerate(np.linspace(a.lo, a.hi, 16)):
            probes.append({"probe_id": f"{axis_id}#{k:02d}", "set": {axis_id: float(v)}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": 20260906, "frames": 10, "dataset": "val/road100",
                                "pipeline": "yolox_s", "probes": probes}, indent=1) + "\n")


def instance_type() -> str | None:
    """EC2 instance type via IMDSv2, or None off EC2."""
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=0.5).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=0.5).read().decode()
    except Exception:
        return None


def fingerprint() -> dict:
    info = cv2.getBuildInformation()
    return {
        "opencv_version": cv2.__version__,
        "cv2_module": getattr(cv2, "__file__", None),
        "build_info_sha256": hashlib.sha256(info.encode()).hexdigest(),
        "kleidicv": "kleidicv" in info.lower(),
        "threads": cv2.getNumThreads(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "instance_type": instance_type(),
    }


def main() -> int:
    from blindspot.runner.dataset import ValidationSet
    from blindspot.runner.registry import load_pipeline
    from blindspot.runner.yolox import COCO_CLASSES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="arm name, e.g. cool-c8g / stock-c8g / stock-c7i")
    parser.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "bench" / "out" / "cool")
    args = parser.parse_args()

    if not args.fixture.is_file():
        make_fixture(args.fixture)
    batch = json.loads(args.fixture.read_text())
    vs = ValidationSet(ROOT / batch["dataset"], COCO_CLASSES)
    frames = vs.load(limit=batch["frames"])
    pipeline = load_pipeline(batch["pipeline"])

    for f in frames[:2]:  # warm-up: dnn graph init and caches, excluded from timing
        pipeline.predict(f.image, f.image_id)

    stage = {"degrade": [], "measure": [], "infer": []}
    results = []
    started = time.perf_counter()
    for probe in batch["probes"]:
        cond = Condition.from_axes(probe["set"])
        preds, truths = [], []
        for f in frames:
            t0 = time.perf_counter()
            img = apply_condition(f.image, cond, seed=batch["seed"])
            t1 = time.perf_counter()
            measure(img)
            t2 = time.perf_counter()
            preds += vs.filter_predictions(pipeline.predict(img, f.image_id))
            t3 = time.perf_counter()
            stage["degrade"].append(t1 - t0)
            stage["measure"].append(t2 - t1)
            stage["infer"].append(t3 - t2)
            truths += f.truths
        results.append({"probe_id": probe["probe_id"], "map50": mean_ap50(preds, truths)["mAP50"]})
    wall = time.perf_counter() - started

    def summary(xs):
        ms = np.array(xs) * 1000.0
        return {"median_ms": round(float(np.median(ms)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3),
                "total_s": round(float(ms.sum() / 1000.0), 3), "n": len(xs)}

    report = {
        "label": args.label,
        "fingerprint": fingerprint(),
        "fixture": str(args.fixture.relative_to(ROOT)),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "frames_per_probe": len(frames),
        "probes": len(batch["probes"]),
        "wall_seconds": round(wall, 3),
        "per_frame": {k: summary(v) for k, v in stage.items()},
        "per_frame_total_median_ms": round(statistics.median(
            (a + b + c) * 1000.0 for a, b, c in zip(stage["degrade"], stage["measure"], stage["infer"])), 3),
        "results": results,
        "command": f"uv run python bench/cool_vs_x86.py --label {args.label}",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    fp = report["fingerprint"]
    print(f"{args.label}: OpenCV {fp['opencv_version']} kleidicv={fp['kleidicv']} "
          f"on {fp['instance_type'] or fp['machine']}")
    for k, v in report["per_frame"].items():
        print(f"  {k:<8} median {v['median_ms']:8.2f} ms  p95 {v['p95_ms']:8.2f} ms")
    print(f"  wall {wall:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
