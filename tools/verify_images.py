"""Verify the worker images the deployed job definitions actually use.

For each Batch job definition (bs-worker-arm64, bs-worker-x86) this reads the
image from ECR and checks, for every layer, the compressed blob against the
manifest digest and the decompressed content against the config's diff_id --
the check Fargate performs when it pulls. It also checks the config's
architecture is the one the job definition runs on.

Found necessary: a legacy-builder amd64 build reused the arm64 base-image
layers under an amd64 config. ECR accepted the push; Fargate refused the pull
with "wrong diff id calculated on extraction".

    AWS_PROFILE=blindspot uv run --group cloud python tools/verify_images.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zlib

import boto3

EXPECTED = {"bs-worker-arm64": "arm64", "bs-worker-x86": "amd64"}
MEDIA = ["application/vnd.oci.image.manifest.v1+json",
         "application/vnd.docker.distribution.manifest.v2+json"]


def main() -> int:
    batch, ecr = boto3.client("batch"), boto3.client("ecr")
    failures = 0
    for name, arch in EXPECTED.items():
        jd = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE")["jobDefinitions"]
        image = max(jd, key=lambda d: d["revision"])["containerProperties"]["image"]
        repo, tag = image.split("/", 1)[1].split(":")

        def blob(digest):
            url = ecr.get_download_url_for_layer(repositoryName=repo, layerDigest=digest)["downloadUrl"]
            return urllib.request.urlopen(url).read()

        manifest = json.loads(ecr.batch_get_image(
            repositoryName=repo, imageIds=[{"imageTag": tag}], acceptedMediaTypes=MEDIA,
        )["images"][0]["imageManifest"])
        config = json.loads(blob(manifest["config"]["digest"]))
        bad = []
        if config.get("architecture") != arch:
            bad.append(f"config architecture {config.get('architecture')} != {arch}")
        for i, (layer, diff_id) in enumerate(zip(manifest["layers"], config["rootfs"]["diff_ids"])):
            data = blob(layer["digest"])
            if "sha256:" + hashlib.sha256(data).hexdigest() != layer["digest"]:
                bad.append(f"layer {i}: compressed digest mismatch")
                continue
            raw = (zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data)
                   if layer["mediaType"].endswith("gzip") else data)
            if "sha256:" + hashlib.sha256(raw).hexdigest() != diff_id:
                bad.append(f"layer {i}: diff_id mismatch (content is not what the config declares)")
        status = "ok" if not bad else "BROKEN"
        print(f"{name}: {arch} image {tag[:12]}, {len(manifest['layers'])} layers -> {status}")
        for b in bad:
            print(f"  {b}")
        failures += bool(bad)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
