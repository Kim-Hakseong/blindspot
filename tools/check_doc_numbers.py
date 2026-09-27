"""Fail the build when a number in the docs no longer matches the benchmark.

Documentation drifts from measurement silently, and a stale figure in a report
is indistinguishable from a fabricated one. So every measured number in a
document carries a machine-checkable citation:

    The Laplacian variance falls to 125.2 <!--bench:degrade_response.axes[7].points[8].measured.laplacian_var-->

The path is `<file-stem>.<dotted path>` into `bench/out/<file-stem>.json`, with
`[i]` for list indices. This tool re-reads the JSON, compares, and exits
non-zero on any mismatch, missing file, or unresolvable path.

    uv run python tools/check_doc_numbers.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_OUT = ROOT / "bench" / "out"

#: `<number> [unit] <!--bench:path-->`, tolerating thousands separators and an
#: optional unit token between the figure and its citation ("37.72 px <!--...-->").
CITATION = re.compile(
    r"(?P<value>-?[\d,]+(?:\.\d+)?)"
    r"\s*(?:[%×xA-Za-zµ°/]{1,12})?\s*"
    r"<!--\s*bench:(?P<path>[^\s>]+?)\s*-->"
)

# Relative tolerance for a cited figure. Documents round; the check should not
# fail because 125.2381 was written as 125.2.
TOLERANCE = 0.005


def _resolve(data, path: str):
    """Walk a dotted path with [i] indices into parsed JSON."""
    node = data
    for token in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if token.startswith("["):
            node = node[int(token[1:-1])]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError(token)
    return node


def check(docs: list[pathlib.Path]) -> list[str]:
    problems: list[str] = []
    cache: dict[str, object] = {}
    checked = 0

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for match in CITATION.finditer(text):
            checked += 1
            raw_path = match.group("path")
            cited = float(match.group("value").replace(",", ""))
            # "frames/frames[0].frame.x" -> bench/out/frames/frames.json, "[0].frame.x"
            head = raw_path.split(".", 1)[0]
            stem = head.split("[", 1)[0]
            rest = raw_path[len(stem):].lstrip(".")
            where = f"{doc.relative_to(ROOT)}:{text[: match.start()].count(chr(10)) + 1}"

            if stem not in cache:
                source = BENCH_OUT / f"{stem}.json"
                if not source.is_file():
                    problems.append(f"{where}: no benchmark output at {source.relative_to(ROOT)}")
                    cache[stem] = None
                    continue
                cache[stem] = json.loads(source.read_text(encoding="utf-8"))

            data = cache[stem]
            if data is None:
                continue

            try:
                actual = float(_resolve(data, rest))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                problems.append(f"{where}: cannot resolve {raw_path!r} ({exc})")
                continue

            denominator = max(abs(actual), 1e-9)
            if abs(actual - cited) / denominator > TOLERANCE:
                problems.append(
                    f"{where}: doc says {cited:g} but {raw_path} is {actual:g}"
                )

    print(f"checked {checked} cited number(s) across {len(docs)} document(s)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    args = parser.parse_args()

    if args.paths:
        docs = args.paths
    else:
        # Only documents that ship. Untracked/ignored working notes are not
        # part of the deliverable and are not checked.
        tracked = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        docs = sorted((ROOT / p) for p in tracked if (ROOT / p).is_file())

    problems = check(docs)
    if problems:
        print("\nstale or unresolvable figures:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("all cited figures match the benchmark output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
