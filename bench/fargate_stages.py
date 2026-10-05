"""x86 vs Graviton on the deployed Batch queues: same image, same probe batch (K2).

Submits `blindspot bench-stages` to bs-arm64 and bs-x86 (2 vCPU / 4 GiB
Fargate tasks each), `--repeats` times per arm, waits, downloads every report
into bench/out/cool/ and writes the comparison:

    AWS_PROFILE=blindspot uv run --group cloud python bench/fargate_stages.py \
        --dataset s3://<bucket>/datasets/road100-<digest> --repeats 3

The dataset prefix is the one `blindspot cloud-run` uploads. Cost per frame is
the measured per-frame time priced at the Fargate task-hour rate in
`cost/ledger.py`; the jobs' own billed time is recorded alongside.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import boto3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.cost.ledger import FARGATE  # noqa: E402
from blindspot.stages import compare_arms  # noqa: E402

VCPU, MEMORY_GB = 2, 4  # infra/blindspot_stack.py WORKER_CPU / WORKER_MEMORY_MIB
OUT = ROOT / "bench" / "out" / "cool"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="required unless --recompute")
    parser.add_argument("--recompute", action="store_true",
                        help="rebuild the comparison from the saved reports; submit nothing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    if args.recompute:
        return summarize(*_load_saved())
    if not args.dataset:
        parser.error("--dataset is required")

    bucket = args.dataset[len("s3://"):].split("/", 1)[0]
    prefix = f"bench/stages-{time.strftime('%Y%m%d-%H%M%S')}"
    batch, s3 = boto3.client("batch"), boto3.client("s3")

    jobs = {}
    for arch in ("arm64", "x86"):
        for r in range(1, args.repeats + 1):
            label = f"fargate-{arch}-r{r}"
            cmd = ["bench-stages", "--label", label, "--dataset", args.dataset,
                   "--frames", str(args.frames), "--out", f"s3://{bucket}/{prefix}"]
            job = batch.submit_job(jobName=f"bs-stages-{arch}-r{r}", jobQueue=f"bs-{arch}",
                                   jobDefinition=f"bs-worker-{arch}",
                                   containerOverrides={"command": cmd},
                                   tags={"project": "blindspot"}, propagateTags=True)
            jobs[job["jobId"]] = (arch, label)
            print(f"submitted {label}")

    pending = set(jobs)
    described = {}
    while pending:
        time.sleep(30)
        for d in batch.describe_jobs(jobs=list(pending))["jobs"]:
            if d["status"] in ("SUCCEEDED", "FAILED"):
                pending.discard(d["jobId"])
                described[d["jobId"]] = d
                print(f"{jobs[d['jobId']][1]}: {d['status']} {d.get('statusReason', '')}")
    failed = [jobs[j][1] for j, d in described.items() if d["status"] != "SUCCEEDED"]
    if failed:
        print(f"failed: {failed}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    arms: dict[str, list[dict]] = {"arm64": [], "x86": []}
    job_seconds = {"arm64": 0.0, "x86": 0.0}
    for job_id, (arch, label) in sorted(jobs.items(), key=lambda kv: kv[1][1]):
        body = s3.get_object(Bucket=bucket, Key=f"{prefix}/{label}.json")["Body"].read()
        report = json.loads(body)
        report["command"] = report["command"] + f" --dataset <road100 on S3>  # Fargate {arch}, {VCPU} vCPU / {MEMORY_GB} GiB"
        (OUT / f"{label}.json").write_text(json.dumps(report, indent=1) + "\n")
        arms[arch].append(report)
        d = described[job_id]
        job_seconds[arch] += (d["stoppedAt"] - d["startedAt"]) / 1000.0

    return summarize(arms, job_seconds, args.repeats, args.frames)


def _load_saved():
    arms: dict[str, list[dict]] = {"arm64": [], "x86": []}
    for f in sorted(OUT.glob("fargate-*-r*.json")):
        report = json.loads(f.read_text())
        arms[report["label"].split("-")[1]].append(report)
    prev = json.loads((OUT / "fargate_x86_vs_arm64.json").read_text())
    return arms, prev["job_running_seconds"], len(arms["arm64"]), arms["arm64"][0]["frames_per_probe"]


def summarize(arms, job_seconds, repeats, frames) -> int:
    rate = {a: VCPU * p["vcpu_h"] + MEMORY_GB * p["gb_h"] for a, p in FARGATE.items()}
    result = compare_arms(arms, rate)
    result.update({
        "task_size": {"vcpu": VCPU, "memory_gb": MEMORY_GB},
        "frames_per_probe": arms["arm64"][0]["frames_per_probe"],
        "probes_per_run": arms["arm64"][0]["probes"],
        "job_running_seconds": job_seconds,
        "command": "uv run --group cloud python bench/fargate_stages.py "
                   f"--dataset <road100 on S3> --repeats {repeats} --frames {frames}",
    })
    (OUT / "fargate_x86_vs_arm64.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=1))
    for arch, a in result["arms"].items():
        print(f"{arch}: {a['cpu_models']} kleidicv={a['kleidicv']} "
              f"{a['mean_ms_per_frame']:.1f} ms/frame (mean) ${a['usd_per_1000_frames']:.5f}/1000 frames")
        for r in a["runs"]:
            print(f"    {r['cpu_model']}: {r['per_frame_total_median_ms']:.1f} ms/frame")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
