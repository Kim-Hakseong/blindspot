"""Three-arm COOL benchmark on EC2 (rule K2): deploy, wait, collect, destroy.

    AWS_PROFILE=blindspot uv run --group cloud python bench/ec2_cool.py --ami <COOL AMI id>

1. Deploys the temporary `BlindspotCoolBench` stack (infra/cool_bench_stack.py)
   with the benchmark pinned to the current commit, which must be on GitHub.
2. Waits for both instances to terminate themselves. Each schedules its own
   shutdown 60 minutes after boot; as a second guard this script terminates
   any instance still alive 65 minutes after the deploy finished.
3. Downloads the reports into bench/out/cool/ and writes the comparison,
   `bench/out/cool/ec2_three_way.json`, with the chip effect (arm 1 vs 2) and
   the COOL effect (arm 2 vs 3) reported separately.
4. Empties the results bucket, destroys the stack, and checks that no
   project=blindspot instance is left in any non-terminated state.

Instance logs go to .cache/cool-bench/ (not committed): they may contain COOL
build output, which is the licensor's confidential information, so only our
own measurements and the version string are published.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import boto3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot.stages import compare_three_way, public_fingerprint  # noqa: E402

STACK = "BlindspotCoolBench"
OUT = ROOT / "bench" / "out" / "cool"
LOGS = ROOT / ".cache" / "cool-bench"
GUARD_MINUTES = 65

#: On-demand Linux, us-east-1, AWS Price List API, read 2026-10-05.
EC2_USD_H = {"c7i.large": 0.08925, "c8g.large": 0.07976, "m8g.4xlarge": 0.71808}
#: COOL software fee per instance type, from the Marketplace offer terms (read
#: 2026-10-05). Zero during the 7-day trial; the list price is used so the
#: figure holds after it.
COOL_USD_H = {"c8g.large": 0.01, "m8g.4xlarge": 0.04}


def arm_table(graviton_type: str, include_x86: bool) -> dict:
    arms = {"x86_stock": ("ec2-x86-stock-", "c7i.large", 0.0)} if include_x86 else {}
    return arms | {"graviton_stock": ("ec2-graviton-stock-", graviton_type, 0.0),
                   "graviton_cool": ("ec2-graviton-cool-", graviton_type, COOL_USD_H[graviton_type])}


def sh(*cmd, cwd=ROOT):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def live_instances(ec2) -> list[dict]:
    res = ec2.describe_instances(Filters=[
        {"Name": "tag:project", "Values": ["blindspot"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]},
    ])
    return [i for r in res["Reservations"] for i in r["Instances"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami", required=True, help="COOL AMI id in us-east-1")
    parser.add_argument("--keep-stack", action="store_true", help="skip the destroy (debugging)")
    parser.add_argument("--graviton-type", default="c8g.large", choices=sorted(COOL_USD_H))
    parser.add_argument("--no-x86", action="store_true", help="arms 2 and 3 only")
    parser.add_argument("--recompute", action="store_true",
                        help="rebuild the comparison from the reports in --out; launch nothing")
    parser.add_argument("--out", type=pathlib.Path, default=OUT,
                        help="where reports go (a smoke test writes outside bench/out)")
    args = parser.parse_args()
    out_dir = args.out
    ARMS = arm_table(args.graviton_type, not args.no_x86)
    if args.recompute:
        return recompute(out_dir, ARMS)
    shape = ["-c", f"graviton_type={args.graviton_type}", "-c", f"include_x86={'0' if args.no_x86 else '1'}"]
    if not re.fullmatch(r"ami-[0-9a-f]{8,17}", args.ami):
        parser.error("not an AMI id")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True)
    if not remote.startswith(commit):
        sys.exit(f"HEAD {commit[:7]} is not origin/main; the instances clone GitHub, so push first")
    digest = __import__("hashlib").sha256((ROOT / "val/road100/manifest.json").read_bytes()).hexdigest()[:12]

    os.environ["JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION"] = "1"
    sh("npx", "cdk", "deploy", STACK, "--require-approval", "never",
       "-c", f"cool_ami={args.ami}", "-c", f"repo_commit={commit}", "-c", f"manifest_digest={digest}", *shape,
       cwd=ROOT / "infra")
    deployed = time.time()

    cfn, ec2, s3 = boto3.client("cloudformation"), boto3.client("ec2"), boto3.client("s3")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in
               cfn.describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]}
    bucket = outputs["ResultsBucket"]
    ids = {role: outputs[key] for role, key in (("x86", "InstanceIdX86"), ("graviton", "InstanceIdGraviton"))
           if key in outputs}

    seen_end: dict[str, float] = {}
    launched: dict[str, dt.datetime] = {}
    meta: dict[str, dict] = {}
    guard_fired = []
    while len(seen_end) < len(ids):
        time.sleep(30)
        desc = ec2.describe_instances(InstanceIds=list(ids.values()))
        for inst in (i for r in desc["Reservations"] for i in r["Instances"]):
            role = next(k for k, v in ids.items() if v == inst["InstanceId"])
            launched[role] = inst["LaunchTime"]
            meta[role] = {"instance_type": inst["InstanceType"], "image_id": inst["ImageId"],
                          "cpu_arch": inst["Architecture"]}
            state = inst["State"]["Name"]
            if state == "terminated" and role not in seen_end:
                seen_end[role] = time.time()
                print(f"{role}: terminated ({inst.get('StateTransitionReason', '')})", flush=True)
            elif state != "terminated" and time.time() - deployed > GUARD_MINUTES * 60:
                print(f"{role}: still {state} after {GUARD_MINUTES} min -> terminating", flush=True)
                ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
                guard_fired.append(role)

    # Collect: reports into the repo, logs and done-markers into .cache only.
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    arms: dict[str, list[dict]] = {k: [] for k in ARMS}
    keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
            for o in p.get("Contents", [])]
    exits = {}
    for key in sorted(keys):
        name = key.rsplit("/", 1)[-1]
        if key.endswith(".log") or name.startswith("done-"):
            s3.download_file(bucket, key, str(LOGS / name))
            if name.startswith("done-"):
                exits[name[5:-5]] = json.loads((LOGS / name).read_text())["exit"]
            continue
        report = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        for arm, (label_prefix, _, _) in ARMS.items():
            if report["label"].startswith(label_prefix):
                report["command"] += "  # bench/ec2_cool.py, EC2 " + ARMS[arm][1]
                # Licence clause 6.1: nothing read from COOL's build is published.
                report["fingerprint"] = public_fingerprint(report["fingerprint"],
                                                           licensed_build=arm == "graviton_cool")
                (out_dir / name).write_text(json.dumps(report, indent=1) + "\n")
                arms[arm].append(report)
    method = None
    log = LOGS / "graviton.log"
    if log.is_file():
        m = re.search(r"^cool_cv2_method=(\S+)", log.read_text(errors="replace"), re.M)
        method = m.group(1) if m else None

    if not args.keep_stack:
        for key in keys:
            s3.delete_object(Bucket=bucket, Key=key)
        sh("npx", "cdk", "destroy", STACK, "--force", "-c", f"cool_ami={args.ami}",
           "-c", f"repo_commit={commit}", *shape, cwd=ROOT / "infra")
    left = live_instances(ec2)
    print(f"project=blindspot instances not terminated: {len(left)}", flush=True)

    ec2_seconds = {r: seen_end[r] - launched[r].timestamp() for r in seen_end}
    run_usd = (ec2_seconds.get("x86", 0) / 3600 * EC2_USD_H["c7i.large"]
               + ec2_seconds.get("graviton", 0) / 3600 * EC2_USD_H[args.graviton_type])
    summary = {"instances": meta, "exit_codes": exits, "guard_terminated": guard_fired,
               "ec2_seconds_observed_upper_bound": ec2_seconds,
               "ec2_usd_observed_upper_bound": run_usd,
               "cool_software_usd": "0 during the 7-day trial",
               "cool_cv2_method": method, "repo_commit": commit, "dataset_manifest": digest,
               "instances_left_running": len(left),
               "command": "uv run --group cloud python bench/ec2_cool.py --ami <COOL AMI id>"
                          + ("" if args.graviton_type == "c8g.large" else f" --graviton-type {args.graviton_type}")
                          + (" --no-x86" if args.no_x86 else "")}
    missing = [a for a, runs in arms.items() if not runs]
    if missing:
        summary["missing_arms"] = missing
        (out_dir / "ec2_three_way.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1), file=sys.stderr)
        return 1
    rates = {a: EC2_USD_H[t] + fee for a, (_, t, fee) in ARMS.items()}
    result = compare_three_way(arms, rates)
    result["arms"]["graviton_cool"].pop("kleidicv", None)
    result.update(summary)
    result["cool_opencv_version"] = arms["graviton_cool"][0]["fingerprint"]["opencv_version"]
    (out_dir / "ec2_three_way.json").write_text(json.dumps(result, indent=1) + "\n")
    show(result)
    return 0 if not left else 2


def recompute(out_dir: pathlib.Path, arms_table: dict) -> int:
    """Rebuild the comparison from saved reports; run facts are kept as recorded."""
    previous = json.loads((out_dir / "ec2_three_way.json").read_text())
    arms = {a: [json.loads(f.read_text()) for f in sorted(out_dir.glob(f"{prefix}r*.json"))]
            for a, (prefix, _, _) in arms_table.items()}
    rates = {a: EC2_USD_H[t] + fee for a, (_, t, fee) in arms_table.items()}
    result = compare_three_way(arms, rates)
    result["arms"]["graviton_cool"].pop("kleidicv", None)
    derived = set(result) | {"arms"}
    result.update({k: v for k, v in previous.items() if k not in derived})
    (out_dir / "ec2_three_way.json").write_text(json.dumps(result, indent=1) + "\n")
    show(result)
    return 0


def show(result: dict) -> None:
    for arm, a in result["arms"].items():
        print(f"{arm}: {a['cpu_models']} OpenCV {a['opencv_version']} "
              f"{a['mean_ms_per_frame']:.1f} ms/frame (mean) ${a['usd_per_1000_frames']:.5f}/1000")
    for eff in ("chip_effect", "cool_effect"):
        if eff not in result:
            continue
        e = result[eff]
        print(f"{eff}: speedup {e['speedup']:.3f} stages "
              + " ".join(f"{s}={v:.3f}" for s, v in e["stage_speedup"].items())
              + f" cost ratio {e['cost_ratio']:.3f} identical mAP {e['map50_identical_probes']}/"
              f"{e['map50_probes_compared']}")


if __name__ == "__main__":
    raise SystemExit(main())
