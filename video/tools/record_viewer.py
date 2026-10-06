"""Records the live report viewer (CloudFront) with real input events.

    uv run --with websockets==15.0.1 python video/tools/record_viewer.py hook
    uv run --with websockets==15.0.1 python video/tools/record_viewer.py tour

hook: the 2-D Blindspot Map. Starts on a passing cell (5.9 ms, 400 lux), dims
      the light one stop, then lengthens the exposure one stop at a time with
      real arrow-key presses on the sliders, crossing the boundary at a
      condition where each axis alone is still inside its one-axis limit.
tour: scrolls the same page through the uncovered-regions and curve panels.
"""

from __future__ import annotations

import asyncio
import sys

from cdp import REPORT_URL, record

READY = ("document.querySelector('canvas.heatmap') !== null && "
         "document.querySelector('.measure-bar') !== null && "
         "document.querySelector('.measure-bar').innerText.includes('mAP')")
SLIDER = "document.querySelectorAll('.sliders input[type=range]')[{}]"
RIGHT = ("ArrowRight", "ArrowRight", 39)


async def step(r, slider: int, hold: float):
    await r.evaluate(SLIDER.format(slider) + ".focus()")
    await r.key(*RIGHT)
    await r.film(hold)


# The four cells the hook visits; filming starts once they have arrived, so the
# initial page load is a cut, not something hidden inside the recording.
HOOK_CELLS = ["x02_y00", "x02_y02", "x04_y02", "x06_y02"]
LOADED = ("(() => { const n = performance.getEntriesByType('resource').filter(e => e.responseEnd > 0)"
          ".map(e => e.name); return %s.every(c => n.some(x => x.includes('/cells/' + c))); })()")


async def hook(r):
    ok = await r.goto(REPORT_URL + "?x=1&y=0", READY)
    print("viewer ready:", ok)
    for _ in range(120):
        if await r.evaluate(LOADED % HOOK_CELLS):
            break
        await asyncio.sleep(0.25)
    await asyncio.sleep(0.5)            # last image decodes and draws
    await r.film(5.0)                   # 5.9 ms, 400 lux: passes
    await step(r, 1, 4.0)               # illuminance 400 -> 173 lux: still passes
    await step(r, 0, 5.0)               # exposure 5.9 -> 10.8 ms: fails, each axis inside its 1-D limit
    await step(r, 0, 7.0)               # exposure 10.8 -> 15.6 ms: the bus called "car"


async def tour(r):
    ok = await r.goto(REPORT_URL, READY)
    print("viewer ready:", ok)
    await asyncio.sleep(1.0)
    await r.film(2.0)
    for _ in range(14):                 # real scroll, a viewport-third at a time
        await r.evaluate("window.scrollBy({top: 300, behavior: 'smooth'})")
        await r.film(0.9)
    await r.film(3.0)


if __name__ == "__main__":
    name = sys.argv[1]
    record({"hook": hook, "tour": tour}[name], name, port=9410 + ["hook", "tour"].index(name))
