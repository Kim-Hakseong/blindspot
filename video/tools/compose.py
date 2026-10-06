"""Final assembly (adapted from Backstop / Standing Order).

- Recorded segments (live viewer over CDP, pty shell sessions) replay at their
  captured timestamps: cuts and end-holds only, never a speed-up. Every cut is
  named in that segment's caption bar.
- The 64 px caption bar on top is an obvious editing overlay naming the source
  of what is on screen (URL or command, and whether it is live, recorded or
  measured). It never imitates browser UI.
- Subtitles are the narration lines, burned into each segment. Narration is Amazon Polly.
- The presenter segment is the entrant's own recording (.cache/video/human/
  intro.mp4). Until it exists a clearly marked placeholder card stands in and
  the output is named *_DRAFT.mp4.

    uv run --with websockets==15.0.1 python video/tools/compose.py
"""

from __future__ import annotations

import json
import subprocess as sp

from cdp import OUT, REPORT_URL, render

CAPH = 64
VW, VH = 1920, 1080 - CAPH
TTS = OUT / "tts"
DUR = json.loads((TTS / "durations.json").read_text())
NARR = json.loads((OUT.parents[1] / "video" / "narration.json").read_text())["lines"]
HUMAN = OUT / "human" / "intro.mp4"

# name, kind, source, cuts [(from, to)], caption, narration [(line, offset)], minimum length
SEGMENTS = [
    ("title", "still", "stills/title.png", None, "title", [("n01", 0.5)], 6.0),
    ("human", "human", None, None, "human", [], 15.0),
    ("hook", "cdp", "hook", None, "viewer", [("n02", 0.3), ("n03", 10.0)], 21.0),
    ("reproduce", "term", "reproduce", None, "reproduce", [("n04", 0.6)], 10.0),
    ("overview", "still", "stills/overview.png", None, "arch", [("n05", 0.5)], 6.0),
    ("results", "still", "stills/results.png", None, "measured", [("n06", 0.5)], 6.0),
    ("tour", "cdp", "tour", None, "viewer", [("n07", 1.0)], 10.0),
    ("agent", "term", "agent", [(0.0, 6.5), (58.6, 66.3)], "agent", [("n08", 0.5)], 8.0),
    ("trace", "term", "trace", None, "trace", [("n09", 0.5)], 8.0),
    ("agentflow", "still", "stills/agentflow.png", None, "arch", [("n10", 0.5)], 6.0),
    ("rules", "term", "rules", None, "rules", [("n11", 1.0)], 8.0),
    ("cool", "still", "stills/cool.png", None, "measured-cool", [("n12", 0.5)], 6.0),
    ("sim2real", "still", "stills/sim2real.png", None, "measured-s2r", [("n13", 0.5)], 6.0),
    ("limits", "still", "stills/limits.png", None, "docs", [("n14", 0.5)], 6.0),
    ("end", "still", "stills/end.png", None, "title", [], 8.0),
]

CAPTIONS = {
    "title": ("Blindspot", "OpenCV 5 · AWS"),
    "human": ("presenter", "recorded by the entrant"),
    "viewer": (REPORT_URL, "live · S3 + CloudFront · real key presses"),
    "reproduce": ("$ uv run blindspot probe …", "recorded shell · real timing"),
    "arch": ("docs/architecture.md", "diagram rendered from text source"),
    "measured": ("bench/out/efficiency.json", "measured"),
    "agent": ("$ blindspot cloud-run --agent", "live AWS + Bedrock · cut: 52 s of model calls"),
    "trace": ("$ tools/show_decisions.py", "live DynamoDB ledger · real timing"),
    "rules": ("$ uv run pytest …", "recorded shell · real timing"),
    "measured-cool": ("bench/out/cool/", "measured on EC2 · instances self-terminated"),
    "measured-s2r": ("bench/out/sim2real.json", "measured · illuminance estimated"),
    "docs": ("docs/technical-report.md §7", "stated limitations"),
}

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600'
         '&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">')
CAP_HTML = FONTS + """<style>html,body{{margin:0;width:1920px;height:64px;background:#12161F;overflow:hidden}}
.bar{{display:flex;align-items:center;justify-content:space-between;height:64px;padding:0 44px;border-bottom:1px solid #252B38;
 font-family:Inter;color:#E8ECF2}}.u{{font-family:'JetBrains Mono';font-size:20px}}
.t{{font-size:14px;letter-spacing:.12em;text-transform:uppercase;color:#A3AEBF;display:flex;align-items:center;gap:10px}}
.dot{{width:9px;height:9px;border-radius:50%;background:#3DD68C}}</style>
<div class="bar"><div class="u">{url}</div><div class="t"><span class="dot"></span>{tag}</div></div>"""
SUB_HTML = FONTS + """<style>html,body{{margin:0;width:1920px;height:120px;background:transparent;overflow:hidden}}
.s{{display:flex;align-items:center;justify-content:center;height:120px;padding:0 140px}}
.x{{font-family:Inter;font-size:27px;font-weight:500;line-height:1.3;color:#E8ECF2;text-align:center;
 background:rgba(11,14,20,.86);padding:10px 24px;border-radius:8px}}</style><div class="s"><div class="x">{text}</div></div>"""
HUMAN_CARD = FONTS + """<style>html,body{{margin:0;width:1920px;height:1016px;background:#0B0E14;color:#E8ECF2;font-family:Inter}}
.c{{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px}}
.k{{font-size:18px;letter-spacing:.14em;text-transform:uppercase;color:#F5B54A}}.h{{font-size:44px;font-weight:600}}</style>
<div class="c"><div class="k">placeholder — not part of the final video</div><div class="h">Presenter segment pending (HUMAN_ACTION)</div></div>"""


def run(*args):
    sp.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def dur(p) -> float:
    return float(sp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                        capture_output=True, text=True, check=True).stdout.strip())


def frames_for(kind, src, cuts):
    raw = json.loads((OUT / f"{'cdp' if kind == 'cdp' else 'term'}_{src}.json").read_text())
    base = OUT / ("cdp" if kind == "cdp" else "term") / src
    frames = [(t, base / name) for t, name in raw]
    if cuts:
        out, offset = [], 0.0
        for a, b in cuts:
            sel = [(t, p) for t, p in frames if a <= t <= b]
            t0 = sel[0][0]
            out += [(offset + t - t0, p) for t, p in sel]
            offset = out[-1][0] + 0.1
        frames = out
    return frames


SCALE = f"scale={VW}:{VH}:force_original_aspect_ratio=decrease,pad={VW}:{VH}:(ow-iw)/2:(oh-ih)/2:color=0x0B0E14,fps=30,format=yuv420p"


def build(name, kind, src, cuts, cap, narration, min_len):
    seg = OUT / "seg" / f"{name}.mp4"
    want = max(min_len, max((off + DUR[l] for l, off in narration), default=0.0) + 0.9)
    if kind == "human" and HUMAN.is_file():
        run("-i", str(HUMAN), "-vf", SCALE, "-an", "-c:v", "libx264", "-crf", "20", str(seg))
    elif kind in ("still", "human"):
        img = OUT / (src if kind == "still" else "stills/human_placeholder.png")
        if kind == "human":
            render(HUMAN_CARD.format(), img, 1920, 1016, port=9460)
        run("-loop", "1", "-t", f"{want:.2f}", "-i", str(img), "-vf", SCALE, "-c:v", "libx264", "-crf", "20", str(seg))
    else:
        frames = frames_for(kind, src, cuts)
        lst = OUT / "seg" / f"{name}.txt"
        lines = [f"file '{p}'\nduration {max(((frames[i + 1][0] - t) if i + 1 < len(frames) else 0.1), 0.02):.3f}"
                 for i, (t, p) in enumerate(frames)] + [f"file '{frames[-1][1]}'"]
        lst.write_text("\n".join(lines))
        hold = max(0.0, want - frames[-1][0])
        vf = SCALE + (f",tpad=stop_mode=clone:stop_duration={hold:.2f}" if hold > 0.01 else "")
        run("-f", "concat", "-safe", "0", "-i", str(lst), "-vf", vf, "-c:v", "libx264", "-crf", "20", str(seg))
    # Subtitles burned per segment (one or two overlays per graph; a single graph
    # with every overlay deadlocked ffmpeg on a long video).
    inputs, chain, last = ["-i", str(seg), "-i", str(OUT / "cap" / f"{cap}.png")], [], "[0:v]"
    for k, (lid, off) in enumerate(narration, 2):
        inputs += ["-i", str(OUT / "cap" / f"sub_{lid}.png")]
        # Bottoms carry the viewer's measure bar and the cards' source and licence
        # footers, so subtitles go to the top except over a terminal session.
        y = "H-h-24" if kind == "term" else "70"
        chain.append(f"{last}[{k}:v]overlay=0:{y}:enable='between(t,{off:.2f},{off + DUR[lid]:.2f})'[s{k}]")
        last = f"[s{k}]"
    chain.append(f"[1:v]{last}vstack=inputs=2,format=yuv420p[out]")
    capped = OUT / "seg" / f"{name}_cap.mp4"
    run(*inputs, "-filter_complex", ";".join(chain), "-map", "[out]", "-c:v", "libx264", "-crf", "20", str(capped))
    return capped


def main():
    for d in ("seg", "cap"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for key, (url, tag) in CAPTIONS.items():
        render(CAP_HTML.format(url=url, tag=tag), OUT / "cap" / f"{key}.png", 1920, CAPH, port=9461)
    for lid, text in NARR.items():
        render(SUB_HTML.format(text=text), OUT / "cap" / f"sub_{lid}.png", 1920, 120, port=9462, transparent=True)

    parts, audio, t = [], [], 0.0
    for name, kind, src, cuts, cap, narration, min_len in SEGMENTS:
        seg = build(name, kind, src, cuts, cap, narration, min_len)
        d = dur(seg)
        for lid, off in narration:
            audio.append((lid, t + off))
        print(f"  {name:<10} {d:5.1f}s @ {t:6.1f}")
        parts.append(seg)
        t += d

    lst = OUT / "seg" / "all.txt"
    lst.write_text("\n".join(f"file '{p}'" for p in parts))
    silent = OUT / "seg" / "silent.mp4"
    run("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent))
    total = dur(silent)

    subbed = silent

    ain, fc = [], []
    for i, (lid, at) in enumerate(audio):
        ain += ["-i", str(TTS / f"{lid}.mp3")]
        fc.append(f"[{i}:a]adelay={int(at * 1000)}|{int(at * 1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(audio)))
    narr = OUT / "seg" / "narration.m4a"
    run(*ain, "-filter_complex", ";".join(fc) + f";{mix}amix=inputs={len(audio)}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,apad[out]",
        "-map", "[out]", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}", str(narr))

    final = OUT / ("blindspot_video.mp4" if HUMAN.is_file() else "blindspot_video_DRAFT.mp4")
    run("-i", str(subbed), "-i", str(narr), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final))
    print(f"FINAL {final.name}: {dur(final):.1f}s")


if __name__ == "__main__":
    main()
