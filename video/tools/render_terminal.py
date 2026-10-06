"""Renders a recorded pty session to frames (adapted from Backstop / Standing Order).

The session is replayed in a styled page at its original timing and captured
over CDP. The text is exactly what the shell produced (ANSI colour and hyperlink
escapes are stripped, bash's job-control notice before the first prompt is
dropped). Colours are the report viewer's tokens.

    uv run --with websockets==15.0.1 python video/tools/render_terminal.py <session> "<kicker>"
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import sys

from cdp import OUT, Chrome, Rec
import websockets

SESSION = sys.argv[1]
KICKER = sys.argv[2] if len(sys.argv) > 2 else "real shell session · recorded over a pty · original timing"
W, H, FPS = 1600, 900, 10

OSC = re.compile(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)")       # hyperlinks etc.
CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ACCOUNT = re.compile(r"(:)\d{12}(?=:)")                     # AWS account id inside an ARN


def chunks(events):
    out, started = [], False
    for t, text in events:
        text = CSI.sub("", OSC.sub("", text)).replace("\r", "")
        text = ACCOUNT.sub(r"\1<account-id>", text)   # redaction, declared in the kicker
        if not started:                         # drop bash's notice before the first prompt
            if "$ " not in text:
                continue
            text, started = text[text.index("$ "):], True
        if text:
            out.append([t, text])
    # End the replay at the last prompt: the recorder's own "exit" is not part of the demo.
    joined, last = "", len(out)
    for i, (_, text) in enumerate(out):
        joined += text
        if joined.endswith("$ "):
            last = i + 1
    return [[t, html.escape(text)] for t, text in out[:last]]


PLAYER = """<!doctype html><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg-0:#0B0E14;--bg-1:#12161F;--border:#252B38;--text-0:#E8ECF2;--text-1:#A3AEBF;--text-2:#6B7684;--pass:#3DD68C;--fail:#FF4D4D;--accent:#4C8DFF}}
html,body{{margin:0;width:{w}px;height:{h}px;background:var(--bg-0);font-family:Inter;overflow:hidden}}
.wrap{{padding:40px 56px}}
.kicker{{font-size:14px;letter-spacing:.12em;text-transform:uppercase;color:var(--text-1);margin-bottom:16px}}
.term{{background:var(--bg-1);border:1px solid var(--border);border-radius:8px;padding:24px 28px;height:{th}px;overflow:hidden}}
#o{{margin:0;font-family:'JetBrains Mono';font-size:17px;line-height:1.5;color:var(--text-0);white-space:pre-wrap;overflow-wrap:anywhere}}
.cur{{display:inline-block;width:9px;height:19px;background:var(--text-0);vertical-align:-3px;animation:b 1s steps(1) infinite}}
@keyframes b{{50%{{opacity:0}}}}
</style>
<div class="wrap"><div class="kicker">{kicker}</div><div class="term" id="t"><pre id="o"></pre></div></div>
<script>
const CH = {chunks};
const o = document.getElementById('o'), t = document.getElementById('t');
const cur = document.createElement('span'); cur.className = 'cur';
function colour(s) {{
  return s.replace(/REFUSED/g, '<span style="color:var(--fail)">REFUSED</span>')
          .replace(/(\\d+ passed)/g, '<span style="color:var(--pass)">$1</span>')
          .replace(/^(\\$ )/gm, '<span style="color:var(--accent)">$1</span>');
}}
window.play = () => CH.forEach(([at, htm]) => setTimeout(() => {{
  cur.remove(); o.insertAdjacentHTML('beforeend', colour(htm)); o.appendChild(cur);
  t.scrollTop = t.scrollHeight; }}, at * 1000));
</script>"""


async def main():
    events = json.loads((OUT / f"pty_{SESSION}.json").read_text())
    ch = chunks(events)
    t0 = ch[0][0]
    ch = [[round(t - t0, 3), h] for t, h in ch]
    span = ch[-1][0] + 2.5
    page = OUT / f"_term_{SESSION}.html"
    page.write_text(PLAYER.format(w=W, h=H, th=H - 130, kicker=html.escape(KICKER), chunks=json.dumps(ch)))
    out = OUT / "term" / SESSION
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    chrome = Chrome(9420, W, H)
    try:
        async with websockets.connect(chrome.ws_url(), max_size=None) as ws:
            r = Rec(ws, out, FPS)
            await r.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await r.goto(page.as_uri(), "document.fonts.status === 'loaded'")
            await asyncio.sleep(0.5)
            await r.evaluate("window.play()")
            await r.film(span)
            frames = r.frames
    finally:
        chrome.close()
    (OUT / f"term_{SESSION}.json").write_text(json.dumps(frames))
    print(f"{SESSION}: {len(frames)} frames, {frames[-1][0]:.1f}s")


asyncio.run(main())
