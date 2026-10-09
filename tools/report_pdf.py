"""Render docs/technical-report.md to PDF for the Devpost file upload.

pandoc turns the Markdown into HTML (citation comments vanish as HTML
comments), headless Chrome prints it. Images resolve relative to docs/.

    uv run python tools/report_pdf.py      # -> dist/Blindspot-technical-report.pdf
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.5;color:#111;max-width:46em;margin:0 auto}
h1{font-size:20pt;margin:0 0 .4em} h2{font-size:14pt;margin:1.4em 0 .4em;border-bottom:1px solid #ccc}
h3{font-size:11.5pt} code{font-family:Menlo,monospace;font-size:9pt;background:#f3f3f3;padding:0 .2em}
pre{background:#f6f6f6;padding:.6em;overflow-x:auto;font-size:8.5pt} img{max-width:100%;page-break-inside:avoid}
table{border-collapse:collapse;font-size:9pt} th,td{border:1px solid #ccc;padding:.25em .5em}
a{color:#0645ad;word-break:break-all}
@page{size:A4;margin:16mm 15mm}
"""


def main() -> int:
    src = ROOT / "docs" / "technical-report.md"
    out = ROOT / "dist" / "Blindspot-technical-report.pdf"
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        css = pathlib.Path(tmp) / "style.css"
        css.write_text(CSS)
        html = ROOT / "docs" / "_technical-report.html"     # beside the images, removed below
        subprocess.run(["pandoc", str(src), "-s", "--metadata", "pagetitle=Blindspot — technical report",
                        "--css", str(css), "-o", str(html)], check=True)
        try:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={out}", html.as_uri()], check=True, capture_output=True, timeout=120)
        finally:
            html.unlink(missing_ok=True)
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
