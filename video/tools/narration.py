"""Narration: each line's numbers are filled from bench/out/*.json, then spoken
by Amazon Polly (generative voice). The video's end card says the voice is
synthetic. Writes video/narration.json (committed) and audio to .cache/video/tts.

    AWS_PROFILE=blindspot uv run --group cloud python video/tools/narration.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import boto3

TOOLS = pathlib.Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
BENCH = REPO / "bench" / "out"
OUT = REPO / ".cache" / "video" / "tts"
VOICE = "Matthew"
AGENT_RUN = "20261006-125326-d307ab"   # the live run recorded in the agent and trace segments


def j(rel):
    return json.loads((BENCH / rel).read_text())


def lines() -> dict[str, str]:
    g, eff, s2r = j("hook_grid.json"), j("efficiency.json"), j("sim2real.json")
    cs, cm = j("cool/ec2_three_way.json"), j("cool/m8g-4xlarge/ec2_three_way.json")
    trace = j(f"cloud_runs/{AGENT_RUN}-decisions.json")
    splits = [(d["accepted_by_scheduler"], int(sum(d["input"]["probes_per_axis"].values())))
              for d in trace if d["tool"] == "reallocate_budget"]
    refused = next(n for ok, n in splits if not ok)
    accepted = next(n for ok, n in splits if ok)
    cores = {k: json.loads((BENCH / f).read_text())["fingerprint"]["logical_cpus"]
             for k, f in (("small", "cool/ec2-graviton-cool-r1.json"), ("large", "cool/m8g-4xlarge/ec2-graviton-cool-r1.json"))}
    pass_cell, dim_cell, fail1, fail2 = g["cells"][0][2], g["cells"][2][2], g["cells"][2][4], g["cells"][2][6]
    car = max((p for p in fail2["detections"]["predictions"] if not p["matched"]), key=lambda p: p["score"]) \
        if fail2.get("detections") else None
    s33 = eff["summary"]["by_grid_level"]["33"]["mean_savings_verified"]
    gap = s2r["gap"]
    return {
        "n01": "Blindspot finds the exact capture conditions where a vision pipeline starts failing, "
               "in physical units, and it reports the conditions your test set never covered.",
        "n02": f"This is the live report. At {pass_cell['exposure_ms']:.1f} milliseconds of exposure and "
               f"{pass_cell['illuminance_lux']:.0f} lux, the YOLOX-S detector passes. Dim the light to "
               f"{dim_cell['illuminance_lux']:.0f} lux. Still passing.",
        "n03": f"Lengthen the exposure to {fail1['exposure_ms']:.1f} milliseconds. Each axis alone is inside its own "
               f"limit, but together, the detector fails. At {fail2['exposure_ms']:.1f} milliseconds it calls a bus a car"
               + (f", with {car['score'] * 100:.0f} percent confidence" if car else "")
               + f". mAP falls from {g['baseline']['mAP50']:.3f} to {fail2['map50']:.3f}.",
        "n04": f"Every number comes with a command that reproduces it. Here it is, recomputed from scratch: "
               f"{fail2['map50']:.3f}, exactly.",
        "n05": "One cdk deploy builds the system. Step Functions runs the search. AWS Batch on Fargate degrades frames "
               "with OpenCV 5 and runs the detector, on Graviton or x86. Every probe is a record in DynamoDB. "
               "The blue boxes are the judgment path. No language model can be imported into it.",
        "n06": f"Every axis has a sharp boundary. Searching instead of sweeping saved {s33:.2f} times the probes at fine "
               f"precision, and nothing at coarse precision. We show both.",
        "n07": "The report leads with what your data never covered, and it lists its own limitations.",
        "n08": "An optional agent on Amazon Bedrock reads the measured boundaries and proposes how to spend the probe "
               "budget. It only proposes.",
        "n09": f"Its first split asked for {refused} probes. After charging its own model calls, the contract could not pay, "
               f"so deterministic code refused it. It proposed {accepted}, which fit. Every call and every verdict is in the ledger.",
        "n10": "Perception, decision, action. The visual measurements drive the agent's proposal, a gate decides, "
               "and anything beyond the budget stops until a person approves it.",
        "n11": "These rules are tests, not promises.",
        "n12": f"We measured COOL, the Graviton-optimized OpenCV, against the stock wheel on the same machine. On our "
               f"workload it was slower: {cs['cool_effect']['speedup']:.2f} on {cores['small']} cores, "
               f"{cm['cool_effect']['speedup']:.2f} on {cores['large']}. Graviton itself ran at {cs['chip_effect']['speedup']:.2f} of x86 speed. We report it as measured.",
        "n13": f"Then we tested our own model against real night photos. The synthetic low-light model is too pessimistic. "
               f"Real photos at an estimated {gap['darkest_bin_lux_median']:.1f} lux still passed, where it predicted failure. "
               f"It over-warns rather than misses failures, and we did not calibrate it on the same photos.",
        "n14": "What this does not show yet: real-world checks for blur, fog and compression, identical results across "
               "different CPUs, and an H.264 axis.",
    }


def main() -> int:
    text = lines()
    (REPO / "video" / "narration.json").write_text(json.dumps({"voice": f"Amazon Polly generative {VOICE}",
                                                                "lines": text}, indent=1) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    polly = boto3.client("polly", region_name="us-east-1")
    durations = {}
    for lid, line in text.items():
        mp3 = OUT / f"{lid}.mp3"
        audio = polly.synthesize_speech(Engine="generative", VoiceId=VOICE, OutputFormat="mp3", Text=line)
        mp3.write_bytes(audio["AudioStream"].read())
        durations[lid] = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
            capture_output=True, text=True, check=True).stdout.strip())
        print(f"{lid} {durations[lid]:.1f}s  {line[:70]}")
    (OUT / "durations.json").write_text(json.dumps(durations, indent=1))
    print(f"total {sum(durations.values()):.1f}s, {sum(len(t) for t in text.values())} characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
