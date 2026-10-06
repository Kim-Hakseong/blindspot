"""Headless Chrome over the DevTools Protocol, for recordings and stills.

Adapted from the Backstop / Standing Order video pipeline. Every frame of a
recording is a real screenshot of a real page at real speed: interactions are
dispatched as genuine input events, waits are filmed, nothing is animated after
the fact. Stills (title, results and caption cards) are HTML rendered here.

Run the tools with `uv run --with websockets==15.0.1 python video/tools/<tool>.py`;
websockets is pinned on the command line so the product's lock file is untouched.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import subprocess
import tempfile
import time
import urllib.request

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VIDEO = pathlib.Path(__file__).resolve().parents[1]
REPO = VIDEO.parent
OUT = REPO / ".cache" / "video"          # frames, segments, audio: never committed
REPORT_URL = "https://d18du1w0ii5yhw.cloudfront.net/"


class Chrome:
    def __init__(self, port: int, w: int, h: int):
        self.port, self.w, self.h = port, w, h
        self.profile = tempfile.mkdtemp(prefix="bs-chrome-")
        self.proc = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={port}",
             f"--user-data-dir={self.profile}", f"--window-size={w},{h}",
             "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run",
             "--no-default-browser-check", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def ws_url(self) -> str:
        for _ in range(100):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json").read())
                return next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("chrome did not start")

    def close(self):
        self.proc.terminate()


class Rec:
    """CDP session; responses matched by message id."""

    def __init__(self, ws, out: pathlib.Path | None = None, fps: int = 10):
        self.ws, self.i, self.out, self.fps = ws, 0, out, fps
        self.frames: list[list] = []
        self.t0 = None

    async def call(self, method, timeout=15, **params):
        self.i += 1
        mid = self.i
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        end = time.time() + timeout
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=max(0.05, end - time.time()))
            except (asyncio.TimeoutError, TimeoutError):
                continue
            m = json.loads(raw)
            if m.get("id") == mid:
                return m.get("result", {})
        return {}

    async def evaluate(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def snap(self):
        if self.t0 is None:
            self.t0 = time.time()
        tick = time.time()
        r = await self.call("Page.captureScreenshot", format="jpeg", quality=85)
        if "data" in r:
            p = self.out / f"f{len(self.frames):05d}.jpg"
            p.write_bytes(base64.b64decode(r["data"]))
            self.frames.append([round(tick - self.t0, 3), p.name])

    async def film(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            tick = time.time()
            await self.snap()
            await asyncio.sleep(max(0, 1 / self.fps - (time.time() - tick)))

    async def key(self, key, code, vk):
        for kind in ("rawKeyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key=key, code=code,
                            windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk)

    async def goto(self, url, ready_js, timeout=45):
        await self.call("Page.navigate", url=url)
        end = time.time() + timeout
        while time.time() < end:
            if await self.evaluate(ready_js):
                return True
            await asyncio.sleep(0.3)
        return False


async def _render(html: str, out: pathlib.Path, w: int, h: int, port: int, transparent: bool):
    chrome = Chrome(port, w, h)
    try:
        async with websockets.connect(chrome.ws_url(), max_size=None) as ws:
            r = Rec(ws)
            await r.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                         deviceScaleFactor=1, mobile=False)
            if transparent:
                await r.call("Emulation.setDefaultBackgroundColorOverride",
                             color={"r": 0, "g": 0, "b": 0, "a": 0})
            page = out.with_suffix(".html")
            page.write_text(html)
            await r.goto(page.as_uri(), "document.fonts.status === 'loaded'")
            await asyncio.sleep(0.6)
            shot = await r.call("Page.captureScreenshot", format="png")
            out.write_bytes(base64.b64decode(shot["data"]))
    finally:
        chrome.close()


def render(html: str, out: pathlib.Path, w: int, h: int, port: int = 9400, transparent=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_render(html, out, w, h, port, transparent))


def record(choreo, name: str, w: int = 1600, h: int = 900, port: int = 9410, fps: int = 10):
    """Run an async choreography against a fresh Chrome and save frames + timing."""
    out = OUT / "cdp" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    async def run():
        chrome = Chrome(port, w, h)
        try:
            async with websockets.connect(chrome.ws_url(), max_size=None) as ws:
                r = Rec(ws, out, fps)
                await r.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                             deviceScaleFactor=1, mobile=False)
                await choreo(r)
                return r.frames
        finally:
            chrome.close()

    frames = asyncio.run(run())
    (OUT / f"cdp_{name}.json").write_text(json.dumps(frames))
    print(f"{name}: {len(frames)} frames, {frames[-1][0]:.1f}s")
