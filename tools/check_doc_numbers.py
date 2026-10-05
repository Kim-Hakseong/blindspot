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

#: Checked on every run whether or not git tracks them yet.
REQUIRED_DOCS = ["docs/devpost-writeup.md"]

#: In these, every decimal figure must carry a citation: the text judges read
#: first may not contain a number the benchmarks did not produce.
STRICT_DOCS = ["docs/devpost-writeup.md"]

DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?!\d|\.\d)")
CODE_OR_COMMENT = re.compile(r"`[^`]*`|<!--.*?-->", re.S)
PLACEHOLDER = re.compile(r"\[(?:TBD\b[^\]]*|update after [^\]]*)\]", re.I)

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


def uncited_decimals(doc: pathlib.Path, text: str) -> list[str]:
    """Decimal figures in a strict document that no citation covers."""
    cited = {m.span("value") for m in CITATION.finditer(text)}
    covered = lambda s, e: any(a <= s and e <= b for a, b in cited)  # noqa: E731
    masked = CODE_OR_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    out = []
    for m in DECIMAL.finditer(masked):
        if not covered(*m.span()):
            line = text[: m.start()].count("\n") + 1
            out.append(f"{doc.relative_to(ROOT)}:{line}: uncited figure {m.group(0)}")
    return out


def placeholders(docs: list[pathlib.Path]) -> int:
    return sum(len(PLACEHOLDER.findall(d.read_text(encoding="utf-8"))) for d in docs)


def check(docs: list[pathlib.Path], strict: set | None = None) -> list[str]:
    strict = set(strict or ())
    problems: list[str] = []
    cache: dict[str, object] = {}
    checked = 0

    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        if doc in strict:
            problems += uncited_decimals(doc, text)
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


def main_with(paths: list[str], submission: bool = False) -> int:
    if paths:
        docs = [pathlib.Path(p).resolve() for p in paths]
    else:
        # Documents that ship, plus required ones even before git tracks them.
        tracked = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        names = sorted(set(tracked) | {d for d in REQUIRED_DOCS if (ROOT / d).is_file()})
        docs = [(ROOT / p) for p in names if (ROOT / p).is_file()]
    strict = {(ROOT / d).resolve() for d in STRICT_DOCS} & {d.resolve() for d in docs}
    docs = [d.resolve() for d in docs]

    problems = check(docs, strict=strict)
    pending = placeholders([d for d in docs if d in strict])
    if problems:
        print("\nstale, unresolvable or uncited figures:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if pending:
        print(f"{pending} placeholder(s) remain in the writeup"
              + (" -- not ready to submit" if submission else ""))
        if submission:
            return 1
    print("all cited figures match the benchmark output")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--submission", action="store_true",
                        help="also fail while any [TBD] / [update after ...] placeholder remains")
    args = parser.parse_args()
    return main_with(args.paths, submission=args.submission)


if __name__ == "__main__":
    raise SystemExit(main())
