"""Fetch only the NOD images the sim-to-real benchmark uses.

NOD (Night Object Detection, BMVC 2021; https://github.com/igor-morawski/NOD)
ships its photos as two zips of 25 and 32 GB on Google Drive, behind a request
form (https://forms.gle/YhHYBofVjphosbeDA) that a person must submit. Its
annotation file declares the images CC BY-NC-SA 2.0; Blindspot uses them as a
non-commercial research benchmark and publishes aggregate metrics only.

This reads each zip's central directory over HTTP range requests and extracts
only the selected images into .cache/nod/images/ (git-ignored), so nothing
close to 57 GB is downloaded and no image enters the repository. Selection,
from the repository's COCO annotations (test and val splits): images with at
least one car whose largest annotated object is not a person (rule C5).

    uv run python tools/fetch_nod.py --drive-file-ids <nikon_zip_id>,<sony_zip_id>
    uv run python tools/fetch_nod.py --verify     # re-check SHA-256 against the manifest

The manifest of names and SHA-256 digests is committed as
bench/fixtures/nod_selection.json; the images are not.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "nod"
MANIFEST = ROOT / "bench" / "fixtures" / "nod_selection.json"
ANNOTATIONS = "https://raw.githubusercontent.com/igor-morawski/NOD/main/annotations/Combined/{}"
SPLITS = ("NOD_Sony_Nikon_test.json", "NOD_Sony_Nikon_val.json")
DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id={}"
BLOCK = 1 << 20


class RemoteFile(io.RawIOBase):
    """A seekable read-only view of a URL that honours Range requests."""

    def __init__(self, url: str):
        self.url, self.pos, self._cache = url, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            if r.status != 206:
                raise IOError(f"server ignored the range request (HTTP {r.status})")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset,
                    io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            lo, hi = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req) as r:
                self._cache[i] = r.read()
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos, BLOCK)
            chunk = self._block(i)[off: off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """One member's bytes with two range requests (local header, then data)."""
    def get(lo: int, hi: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        for wait in (10, 30, 90, 0):  # Drive answers 503 / 429 under load
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except urllib.error.HTTPError as err:
                if err.code not in (429, 500, 502, 503) or not wait:
                    raise
                time.sleep(wait)
    head = get(info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    data = get(start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return data


def selection() -> dict[str, dict]:
    chosen = {}
    for split in SPLITS:
        path = CACHE / split
        if not path.is_file():
            CACHE.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(ANNOTATIONS.format(split), path)
        d = json.loads(path.read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_image: dict[int, list] = {}
        for a in d["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            anns = by_image.get(im["id"], [])
            if not any(cats[a["category_id"]] == "car" for a in anns):
                continue
            if cats[max(anns, key=lambda a: a["area"])["category_id"]] == "person":
                continue  # rule C5: a person is the main subject
            chosen[im["file_name"]] = {
                "split": split, "width": im["width"], "height": im["height"],
                "boxes": [{"cls": cats[a["category_id"]], "xywh": a["bbox"]} for a in anns],
            }
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-file-ids", help="comma-separated Drive ids of the NOD zips")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    images = CACHE / "images"

    if args.verify:
        manifest = json.loads(MANIFEST.read_text())
        bad = [n for n, e in manifest["files"].items()
               if hashlib.sha256((images / n).read_bytes()).hexdigest() != e["sha256"]]
        print(f"{len(manifest['files']) - len(bad)} ok, {len(bad)} mismatched or missing")
        return 1 if bad else 0

    wanted = selection()
    images.mkdir(parents=True, exist_ok=True)
    files = {}
    for file_id in args.drive_file_ids.split(","):
        remote = RemoteFile(DRIVE.format(file_id))
        with zipfile.ZipFile(remote) as z:
            members = {pathlib.Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
            hits = [n for n in wanted if n in members]
            print(f"zip {file_id[:6]}…: {len(members)} members, {len(hits)} selected")
            url = DRIVE.format(file_id)

            def one(n):
                out = images / n
                # A file left by an interrupted run is refetched unless its CRC matches.
                if not out.is_file() or zlib.crc32(out.read_bytes()) != members[n].CRC:
                    tmp = out.with_suffix(out.suffix + ".part")
                    tmp.write_bytes(fetch_member(url, members[n]))
                    tmp.replace(out)
                return n, {"zip_member": members[n].filename,
                           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
            with ThreadPoolExecutor(8) as pool:
                for i, (n, entry) in enumerate(pool.map(one, hits), 1):
                    files[n] = entry
                    if i % 25 == 0:
                        print(f"  {i}/{len(hits)}", flush=True)
    missing = sorted(set(wanted) - set(files))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "dataset": "NOD (Night Object Detection), Morawski et al., BMVC 2021",
        "repository": "https://github.com/igor-morawski/NOD",
        "licence": "CC BY-NC-SA 2.0 (as declared in the NOD annotation files)",
        "use": "non-commercial research benchmark; images not redistributed",
        "selection": "test+val splits; >=1 car; largest annotated object not a person",
        "files": dict(sorted(files.items())), "missing": missing,
    }, indent=1) + "\n")
    print(f"{len(files)} images extracted, {len(missing)} not found in the zips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
