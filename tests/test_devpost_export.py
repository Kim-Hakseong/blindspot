"""The Devpost text is generated from docs/devpost-writeup.md, never edited apart."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from devpost_export import BUILT_WITH, export  # noqa: E402

WRITEUP = """## Inspiration
Why. 0.611 <!--bench:efficiency.baseline.mAP50--> mAP.

## What it does
It finds boundaries.

## How I built it
- OpenCV 5.

## Results (measured, `bench/out/*.json`)
- 4.12× <!--bench:x.y--> savings.

## Challenges
Hard.

## What I learned
Lots.

## What's next
More.

## Known limitations
- Some.
"""


def test_citations_are_stripped_and_numbers_kept():
    out = export(WRITEUP)
    assert "<!--" not in out and "0.611 mAP" in out and "4.12× savings" in out


def test_sections_follow_devposts_fields():
    out = export(WRITEUP)
    order = ["## Inspiration", "## What it does", "## How I built it", "## Challenges I ran into",
             "## Accomplishments that I'm proud of", "## What I learned", "## What's next for Blindspot"]
    positions = [out.index(h) for h in order]
    assert positions == sorted(positions)
    # results become the accomplishments; limitations stay attached to them
    acc = out[out.index("## Accomplishments"):out.index("## What I learned")]
    assert "savings" in acc and "Known limitations" in acc


def test_built_with_respects_devposts_limit():
    assert 0 < len(BUILT_WITH) <= 25 and len(set(BUILT_WITH)) == len(BUILT_WITH)
