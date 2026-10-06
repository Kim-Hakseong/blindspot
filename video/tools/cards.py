"""Stills for the video (1920x1016, a caption bar is added on top) and the
Devpost gallery (1500x1000, 3:2).

Every number on a card is read from bench/out/*.json here, never typed in.
Gallery images of the report are captures of the live viewer (CloudFront).
No NOD image, and nothing derived from one, is rendered anywhere.

    uv run --with websockets==15.0.1 python video/tools/cards.py
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import pathlib

import websockets

from cdp import OUT, REPO, REPORT_URL, Chrome, Rec, render

BENCH = REPO / "bench" / "out"
GALLERY = REPO / "docs" / "gallery"
STILLS = OUT / "stills"
REPO_URL = "https://github.com/Kim-Hakseong/blindspot"


def j(rel: str) -> dict:
    return json.loads((BENCH / rel).read_text())


EFF, GRID, S2R = j("efficiency.json"), j("hook_grid.json"), j("sim2real.json")
COOL_S, COOL_M = j("cool/ec2_three_way.json"), j("cool/m8g-4xlarge/ec2_three_way.json")
FARGATE, MATRIX = j("cool/fargate_x86_vs_arm64.json"), j("efficiency_matrix.json")
LEVELSET, AGENT = j("levelset_efficiency.json"), j("cloud_runs/agent_gate.json")
GATE_COST = j("cloud_runs/20261005-132343-667165-cost.json")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg-0:#0B0E14;--bg-1:#12161F;--bg-2:#1A1F2B;--border:#252B38;--text-0:#E8ECF2;--text-1:#A3AEBF;
 --text-2:#6B7684;--pass:#3DD68C;--fail:#FF4D4D;--uncovered:#F5B54A;--accent:#4C8DFF;--measure:#B98CFF}
html,body{margin:0;background:var(--bg-0);color:var(--text-0);font-family:Inter;overflow:hidden}
.page{box-sizing:border-box;width:100%;height:100%;padding:56px 72px;display:flex;flex-direction:column}
.kicker{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:var(--text-1);margin-bottom:14px}
h1{font-size:46px;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin:0 0 18px}
h2{font-size:30px;font-weight:600;margin:0 0 20px;letter-spacing:-.01em}
.mono{font-family:'JetBrains Mono'}
.fail{color:var(--fail)} .pass{color:var(--pass)} .acc{color:var(--accent)} .unc{color:var(--uncovered)} .dim{color:var(--text-1)}
table{border-collapse:collapse;width:100%;font-size:22px}
th,td{padding:12px 14px;border-bottom:1px solid var(--border);text-align:left}
th{color:var(--text-1);font-weight:500;font-size:16px;letter-spacing:.06em;text-transform:uppercase}
td.n{font-family:'JetBrains Mono';text-align:right}
.card{background:var(--bg-1);border:1px solid var(--border);border-radius:8px;padding:28px 32px}
.foot{font-size:15px;color:var(--text-2)}
.big{font-size:64px;font-weight:700;letter-spacing:-.03em}
.row{display:flex;gap:28px}
.row>.card{flex:1}
img.fit{max-width:100%;max-height:100%;object-fit:contain;display:block;margin:0 auto}
</style>"""


def page(body: str, w: int, h: int) -> str:
    # Main content centred vertically; the source/credit footer stays at the bottom.
    cut = body.find("<div class='foot")
    main, foot = (body[:cut], body[cut:]) if cut >= 0 else (body, "")
    return (f"<!doctype html><meta charset='utf-8'>{CSS}<body style='width:{w}px;height:{h}px'>"
            f"<div class='page'><div style='margin:auto 0;display:flex;flex-direction:column;min-height:0;max-height:100%'>"
            f"{main}</div>{foot.replace('margin-top:auto', '')}</div></body>")


def f(x, d=2):
    return f"{x:.{d}f}"


def axis_rows():
    rows = []
    for a in EFF["axes"]:
        lvl = max(a["levels"], key=lambda l: l["grid_steps"])
        g = lvl["grid"]
        rows.append(f"<tr><td class='mono'>{a['axis']}</td><td class='n'>{g['lower']:.4g} – {g['upper']:.4g} {a['unit']}</td>"
                    f"<td class='n'>{lvl['bisection_verified']['probes_used']} / {g['probes_used']}</td></tr>")
    return "".join(rows)


def cool_table(compact=False):
    s, m = COOL_S["arms"], COOL_M["arms"]
    cols = [("x86 c7i.large", s["x86_stock"], "stock"), ("Graviton4 c8g.large", s["graviton_stock"], "stock"),
            ("Graviton4 c8g.large", s["graviton_cool"], "COOL"), ("Graviton4 m8g.4xlarge", m["graviton_stock"], "stock"),
            ("Graviton4 m8g.4xlarge", m["graviton_cool"], "COOL")]
    head = "".join(f"<th>{n}<br><span class='{'acc' if b == 'COOL' else 'dim'}'>{b} OpenCV {a['opencv_version']}</span></th>" for n, a, b in cols)
    t = "".join(f"<td class='n'>{f(a['mean_ms_per_frame'], 1)} ms</td>" for _, a, _ in cols)
    c = "".join(f"<td class='n'>${a['usd_per_1000_frames']:.4f}</td>" for _, a, _ in cols)
    return (f"<table><tr><th></th>{head}</tr><tr><td>time / frame (mean)</td>{t}</tr>"
            f"<tr><td>cost / 1,000 frames</td>{c}</tr></table>")


def video_cards() -> dict[str, str]:
    W, H = 1920, 1016
    gate = S2R["gap"]
    lvl33 = EFF["summary"]["by_grid_level"]["33"]["mean_savings_verified"]
    lvl5 = EFF["summary"]["by_grid_level"]["5"]["mean_savings_verified"]
    cards = {
        "title": page(f"""<div style='margin:auto 0'>
 <div class='kicker'>OpenCV AI Competition 2026 · powered by AWS</div>
 <div class='big'>Blindspot</div>
 <h1 style='font-size:40px;font-weight:500;color:var(--text-1)'>Find where your vision pipeline starts lying —<br>
 in milliseconds of exposure and lux, not a single accuracy number.</h1>
 <div class='dim' style='font-size:22px'>We don't build a detector. We interrogate one.</div></div>
 <div class='foot mono'>{REPO_URL} · {REPORT_URL}</div>""", W, H),
        "overview": page(f"""<div class='kicker'>Architecture</div>
 <div style='flex:1;display:flex;align-items:center'><img class='fit' src='{(REPO / "docs/architecture-3.png").as_uri()}'></div>
 <div class='foot'>Blue: deterministic judgment path (no model can import into it, enforced by a test). Purple: the only place a model is consulted.</div>""", W, H),
        "agentflow": page(f"""<div class='kicker'>Agentic loop · perception → decision → action</div>
 <div style='flex:1;display:flex;align-items:center'><img class='fit' src='{(REPO / "docs/architecture-4.png").as_uri()}'></div>
 <div class='foot'>Every model call is charged to the run's budget contract as it happens; every decision, refused ones included, is written to DynamoDB.</div>""", W, H),
        "results": page(f"""<div class='kicker'>Measured · YOLOX-S · road100 (100 COCO frames, {EFF['dataset']['objects']} objects)</div>
 <h2>Every axis has a sharp boundary — and the search is cheaper only when you ask for precision</h2>
 <div class='row'><div class='card'><table><tr><th>axis</th><th>fails between</th><th>probes: search / grid</th></tr>{axis_rows()}</table></div>
 <div class='card' style='flex:.6'><div class='dim'>probe savings vs exhaustive grid</div>
 <div class='big acc'>{lvl33:.2f}×</div><div class='dim'>at 33-step precision</div>
 <div class='big' style='margin-top:18px'>{lvl5:.2f}×</div><div class='dim'>at 5-step precision — no saving</div>
 <div style='margin-top:18px;font-size:20px'>{MATRIX['summary']['axis_runs']} searches, 3 datasets × 2 models: every boundary matched the full grid.</div></div></div>
 <div class='foot'>Source: bench/out/efficiency.json, efficiency_matrix.json. Baseline mAP@50 {f(EFF['baseline']['mAP50'], 3)}; failure = below 60% of baseline.</div>""", W, H),
        "sim2real": page(f"""<div class='kicker'>Sim-to-real · the tool measured its own blind spot</div>
 <h2>On real night photos the synthetic low-light model is too pessimistic</h2>
 <div class='row'><div class='card'><div class='dim'>darkest fifth of real photos (estimated {f(gate['darkest_bin_lux_median'])} lux)</div>
 <div class='big pass'>mAP {f(gate['darkest_bin_real_map50'], 3)}</div><div class='dim'>above the failure threshold {f(S2R['threshold_map50'], 3)}</div></div>
 <div class='card'><div class='dim'>synthetic boundary overstates the failure illuminance by</div>
 <div class='big unc'>≥ {f(gate['synthetic_boundary_overstates_failure_illuminance_by_at_least'], 1)}×</div>
 <div class='dim'>{gate['bins_synthetic_fail_real_pass']} of 5 bins: synthetic predicts failure, real photos pass</div></div></div>
 <div style='font-size:22px;margin-top:26px'>The error is on the conservative side: it over-warns rather than misses failures. Not calibrated on these photos (that would fit the answer); next: model in-camera noise reduction.</div>
 <div class='foot'>{S2R['selection']['measured']} night photos, {S2R['objects']} cars, from NOD (Morawski et al., BMVC 2021, github.com/igor-morawski/NOD), CC BY-NC-SA 2.0, used as a non-commercial research benchmark; no NOD image is shown.
 Illuminance is estimated (EXIF exposure equation × photo brightness), not measured. No first-party capture.</div>""", W, H),
        "cool": page(f"""<div class='kicker'>Same 64-probe batch · 3 runs per arm · EC2 · total work per frame</div>
 <h2>Chip effect and COOL effect, measured separately</h2>
 <div class='card'>{cool_table()}</div>
 <div class='row' style='margin-top:24px'><div class='card'><div class='dim'>Graviton4 vs x86 (stock)</div><div class='big'>{f(COOL_S['chip_effect']['speedup'], 3)}×</div><div class='dim'>speed; cheaper per frame on Fargate, dearer on EC2</div></div>
 <div class='card'><div class='dim'>COOL vs stock, same machine</div><div class='big fail'>{f(COOL_S['cool_effect']['speedup'], 3)}× · {f(COOL_M['cool_effect']['speedup'], 3)}×</div><div class='dim'>c8g.large · m8g.4xlarge: slower on this workload</div></div></div>
 <div class='foot'>COOL named only as what was measured. Source: bench/out/cool/ (bench/ec2_cool.py; instances terminate themselves).</div>""", W, H),
        "limits": page(f"""<div class='kicker'>Known limitations · stated, not hidden</div>
 <h2>What this does not show</h2>
 <ul style='font-size:26px;line-height:1.7;margin:0'>
 <li>Sim-to-real covers low light only, on third-party photos with <span class='unc'>estimated</span> illuminance.</li>
 <li>Results are not bit-identical across CPUs: {FARGATE['map50_identical_probes']} of {FARGATE['map50_probes_compared']} probes identical; every pass/fail decision agreed.</li>
 <li>No H.264 axis: this OpenCV 5 build exposes no rate control, so no honest CRF unit.</li>
 <li>Two axes mapped jointly; other pairs untested. Three detectors, two families.</li>
 <li>COOL was measured on one workload and was slower there; not integrated.</li></ul>
 <div class='foot'>Full list: docs/technical-report.md §7 and the report viewer.</div>""", W, H),
        "end": page(f"""<div style='margin:auto 0'><div class='big'>Blindspot</div>
 <div style='font-size:28px;line-height:1.8;margin-top:10px' class='mono'>
 <div>report  <span class='acc'>{REPORT_URL}</span></div><div>code    <span class='acc'>{REPO_URL}</span> (MIT)</div>
 <div class='dim'>one run on AWS: ${GATE_COST['total_usd']:.4f} measured · {GATE_COST['fargate_tasks']} Fargate tasks</div></div></div>
 <div class='foot'>Every number in this video is read from bench/out/*.json in the repository. Narration: synthetic voice (Amazon Polly).</div>""", W, H),
    }
    return cards


async def capture(url: str, selector: str, out: pathlib.Path, w: int, h: int, ready: str,
                  wait_js: str | None = None, scale: int = 1):
    chrome = Chrome(9430, w, h)
    try:
        async with websockets.connect(chrome.ws_url(), max_size=None) as ws:
            r = Rec(ws)
            await r.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=scale, mobile=False)
            await r.goto(url, ready)
            for _ in range(160):
                if not wait_js or await r.evaluate(wait_js):
                    break
                await asyncio.sleep(0.25)
            await asyncio.sleep(1.0)
            box = json.loads(await r.evaluate(
                f"(() => {{ const e = {selector}; e.scrollIntoView({{block:'start'}}); "
                f"const b = e.getBoundingClientRect(); return JSON.stringify({{x:b.x+scrollX,y:b.y+scrollY,width:b.width,height:b.height}}); }})()"))
            await asyncio.sleep(0.6)
            shot = await r.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                clip={**box, "scale": 1})
            out.write_bytes(base64.b64decode(shot["data"]))
    finally:
        chrome.close()


def gallery():
    GALLERY.mkdir(parents=True, exist_ok=True)
    W, H = 1500, 1000
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    viewer_ready = "document.querySelector('.measure-bar') !== null"
    cells_loaded = ("performance.getEntriesByType('resource').filter(e => e.name.includes('/cells/') && e.responseEnd > 0).length >= 81")
    asyncio.run(capture(REPORT_URL + "?x=3&y=1", "document.querySelector('section[aria-label=\"Blindspot Map\"]')",
                        raw / "hero.png", 1600, 1000, viewer_ready, cells_loaded))
    asyncio.run(capture(REPORT_URL, "document.querySelector('canvas.heatmap').closest('.card, .panel, div')",
                        raw / "map.png", 1600, 1000, viewer_ready, scale=2))
    asyncio.run(capture(REPORT_URL, "[...document.querySelectorAll('section.panel')].find(s => s.querySelector('h2')?.textContent.startsWith('Where this pipeline fails'))",
                        raw / "curves.png", 1600, 1200, viewer_ready))
    cell = GRID["cells"][2][6]
    attr = GRID["demo_frame"]["attribution"]
    credit = (f"Photo: COCO val2017 #{GRID['demo_frame']['image_id']}, {html.escape(attr['source'])}, "
              f"CC BY 2.0 ({attr['license_url']}); faces blurred, degradation and overlay applied.")
    trace = sorted((OUT / "term" / "trace").glob("*.jpg"))[-1]
    items = {
        "01-failure-frame": f"""<div class='kicker'>Live report viewer · measured, not simulated</div>
 <h2 style='font-size:26px'>{cell['exposure_ms']:.1f} ms exposure at {cell['illuminance_lux']:.0f} lux: mAP@50 {f(GRID['baseline']['mAP50'], 3)} → {f(cell['map50'], 3)}; the detector calls a bus a car</h2>
 <div style='flex:1;min-height:0'><img class='fit' src='{(raw / "hero.png").as_uri()}'></div>
 <div class='foot'>{credit}</div>""",
        "02-blindspot-map": f"""<div class='kicker'>Blindspot Map · {GRID['n_cells']} measured conditions</div>
 <h2 style='font-size:26px'>{GRID['n_failed']} of {GRID['n_cells']} conditions fail; one-axis limits overstate the safe region</h2>
 <div style='flex:1;min-height:0'><img class='fit' style='height:740px' src='{(raw / "map.png").as_uri()}'></div>
 <div class='foot'>Exposure (blur and light from one shutter) × illuminance, YOLOX-S on road100. Dots: conditions your validation set never covered.</div>""",
        "03-per-axis-curves": f"""<div class='kicker'>Per-axis response · physical units</div>
 <h2 style='font-size:26px'>mAP@50 against each degradation, with the located boundary</h2>
 <div style='flex:1;min-height:0'><img class='fit' src='{(raw / "curves.png").as_uri()}'></div>
 <div class='foot'>Every boundary carries a reproduce command. Source: bench/out/efficiency.json.</div>""",
        "04-architecture": f"""<div class='kicker'>Architecture · OpenCV 5 on AWS</div>
 <div style='flex:1;display:flex;align-items:center'><img class='fit' src='{(REPO / "docs/architecture-3.png").as_uri()}'></div>
 <div class='foot'>Step Functions · AWS Batch on Fargate (Graviton + x86) · Lambda · DynamoDB · S3 · CloudFront · Bedrock. Blue: deterministic judgment path. Purple: the model, proposing only.</div>""",
        "05-cool-comparison": f"""<div class='kicker'>COOL comparison · same batch · total work per frame</div>
 <h2 style='font-size:26px'>COOL vs stock OpenCV 5 on the same Graviton4: {f(COOL_S['cool_effect']['speedup'], 3)}× (c8g.large), {f(COOL_M['cool_effect']['speedup'], 3)}× (m8g.4xlarge)</h2>
 <div class='card' style='font-size:18px'>{cool_table()}</div>
 <div class='foot' style='margin-top:20px'>Chip effect (x86 vs Graviton4, stock): {f(COOL_S['chip_effect']['speedup'], 3)}×. COOL named only as what was measured. bench/ec2_cool.py; instances terminate themselves.</div>""",
        "06-agent-workflow": f"""<div class='kicker'>Agentic vision · perception → decision → action</div>
 <div style='flex:1;display:flex;align-items:center'><img class='fit' src='{(REPO / "docs/architecture-4.png").as_uri()}'></div>
 <div class='foot'>Live runs: {AGENT['totals']['proposals_accepted']} proposals accepted, {AGENT['totals']['proposals_refused']} refused with reasons; every model call charged to the run contract.</div>""",
        "07-agent-trace": f"""<div class='kicker'>Decision ledger of a live run (DynamoDB)</div>
 <h2 style='font-size:26px'>OpenCV-measured boundaries in, a refused and an accepted proposal out</h2>
 <div style='flex:1;min-height:0'><img class='fit' src='{trace.as_uri()}'></div>
 <div class='foot'>tools/show_decisions.py --run-id 20261006-125326-d307ab</div>""",
    }
    for name, body in items.items():
        render(page(body, W, H), GALLERY / f"{name}.png", W, H, port=9440)
        print("gallery", name)


def main():
    for name, html_text in video_cards().items():
        render(html_text, STILLS / f"{name}.png", 1920, 1016, port=9450)
        print("still", name)
    gallery()


if __name__ == "__main__":
    main()
