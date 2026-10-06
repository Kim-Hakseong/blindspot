"""Render the Mermaid diagrams in docs/architecture.md to PNG (rule D2).

The text in docs/architecture.md is the source of truth; this regenerates the
images from it with a pinned Mermaid in headless Chrome.

    uv run python tools/render_architecture.py
"""

from __future__ import annotations

import html
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/mermaid.min.js"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SIZES = [(1700, 2600), (1400, 1100), (2200, 1100), (2200, 1100)]


def _trim(path: pathlib.Path, pad: int = 24) -> None:
    """Crop the empty page background around the rendered diagram."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    bg = img[0, 0].astype(int)
    mask = np.any(np.abs(img.astype(int) - bg) > 8, axis=2)
    ys, xs = np.nonzero(mask)
    if len(ys):
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        cv2.imwrite(str(path), img[y0:y1, x0:x1])


def main() -> int:
    source = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", source, flags=re.S)
    for n, (block, (w, h)) in enumerate(zip(blocks, SIZES), 1):
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<script src="{MERMAID}"></script>
<style>body{{margin:24px;background:#0B0E14}}</style></head><body>
<pre class="mermaid">{html.escape(block)}</pre>
<script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script></body></html>"""
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
            fh.write(page)
        out = ROOT / "docs" / f"architecture-{n}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={w},{h}", "--virtual-time-budget=15000",
                        f"--screenshot={out}", f"file://{fh.name}"],
                       check=True, capture_output=True)
        _trim(out)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
