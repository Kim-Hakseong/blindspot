"use client";

// Hero: the Blindspot Map and the frame it was measured on.
//
// Every frame shown was actually scored: sliders snap to the cells for which
// the probe saved its degraded frame and predictions, and the viewer never
// interpolates, re-renders or animates between them.

import { useEffect, useMemo, useRef, useState } from "react";
import { boundaryEdges, fmt, imageStops, loadOrder, reproduceCommand, uncoveredMask, viridis } from "../lib/map.mjs";
import type { Cell, Map2D, Report } from "../lib/types";

const FRAME_W = 640;
const FRAME_H = 480;
const MAP_W = 520;
const MAP_H = 480;
const PAD = { left: 64, right: 8, top: 8, bottom: 48 };

function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Each frame is shown as soon as it arrives, nearest the selected cell first.
// (Waiting for all 81 left the panel on "loading" for 15 s on a cold cache.)
function useImages(map: Map2D, i: number, j: number) {
  const [images, setImages] = useState<Record<string, HTMLImageElement>>({});
  const start = useRef({ i, j });
  useEffect(() => {
    for (const name of loadOrder(map.cells, start.current.i, start.current.j)) {
      const img = new Image();
      img.onload = () => setImages((prev) => ({ ...prev, [name]: img }));
      img.src = `data/cells/${name}`;
    }
  }, [map]);
  return images;
}

function drawFrame(ctx: CanvasRenderingContext2D, img: HTMLImageElement | undefined, cell: Cell) {
  ctx.fillStyle = token("--bg-2");
  ctx.fillRect(0, 0, FRAME_W, FRAME_H);
  if (!img) {
    ctx.fillStyle = token("--text-2");
    ctx.font = `13px ${token("--font-mono")}, monospace`;
    ctx.fillText("loading measured frame…", 16, 24);
    return;
  }
  const s = Math.min(FRAME_W / img.width, FRAME_H / img.height);
  const ox = (FRAME_W - img.width * s) / 2;
  const oy = (FRAME_H - img.height * s) / 2;
  ctx.drawImage(img, ox, oy, img.width * s, img.height * s);
  const det = cell.detections;
  if (!det) return;

  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = token("--text-2");
  for (const t of det.truths) {
    const [x0, y0, x1, y1] = t.box;
    ctx.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
  }
  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.font = `13px ${token("--font-mono")}, monospace`;
  for (const p of det.predictions.filter((d) => d.score >= 0.5)) {
    const [x0, y0, x1, y1] = p.box;
    const color = p.matched ? token("--pass") : token("--fail");
    ctx.strokeStyle = color;
    ctx.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
    const label = `${p.label} ${p.score.toFixed(2)} ${p.matched ? "ok" : "miss"}`;
    const w = ctx.measureText(label).width + 8;
    ctx.fillStyle = token("--bg-0");
    ctx.globalAlpha = 0.8;
    ctx.fillRect(ox + x0 * s, oy + y0 * s - 18, w, 18);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.fillText(label, ox + x0 * s + 4, oy + y0 * s - 5);
  }
}

function drawMap(ctx: CanvasRenderingContext2D, map: Map2D, baseline: number,
                 uncovered: boolean[][], current: Cell) {
  const nx = map.x.values.length;
  const ny = map.y.values.length;
  const cw = (MAP_W - PAD.left - PAD.right) / nx;
  const ch = (MAP_H - PAD.top - PAD.bottom) / ny;
  const X = (i: number) => PAD.left + i * cw;
  const Y = (j: number) => PAD.top + j * ch;

  ctx.fillStyle = token("--bg-2");
  ctx.fillRect(0, 0, MAP_W, MAP_H);

  for (const row of map.cells) for (const c of row) {
    if (c.failed) {
      ctx.fillStyle = token("--fail-dim");
      ctx.fillRect(X(c.i), Y(c.j), cw, ch);
      // Hatch: failure is marked by pattern as well as colour.
      ctx.save();
      ctx.beginPath(); ctx.rect(X(c.i), Y(c.j), cw, ch); ctx.clip();
      ctx.strokeStyle = token("--fail"); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      for (let k = -ch; k < cw; k += 6) {
        ctx.beginPath(); ctx.moveTo(X(c.i) + k, Y(c.j) + ch); ctx.lineTo(X(c.i) + k + ch, Y(c.j)); ctx.stroke();
      }
      ctx.restore();
    } else {
      ctx.fillStyle = viridis(c.map50 / baseline);
      ctx.fillRect(X(c.i), Y(c.j), cw, ch);
    }
    if (uncovered[c.j][c.i]) {
      ctx.fillStyle = token("--uncovered");
      // Uncovered is meant to be the most visible mark on the map (Design.md).
      ctx.globalAlpha = 0.9;
      for (let dx = 4; dx < cw - 1; dx += 8) for (let dy = 4; dy < ch - 1; dy += 8) {
        ctx.fillRect(X(c.i) + dx, Y(c.j) + dy, 2.5, 2.5);
      }
      ctx.globalAlpha = 1;
    }
  }

  ctx.strokeStyle = token("--border");
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= nx; i++) { ctx.beginPath(); ctx.moveTo(X(i), Y(0)); ctx.lineTo(X(i), Y(ny)); ctx.stroke(); }
  for (let j = 0; j <= ny; j++) { ctx.beginPath(); ctx.moveTo(X(0), Y(j)); ctx.lineTo(X(nx), Y(j)); ctx.stroke(); }

  ctx.strokeStyle = token("--fail");
  ctx.lineWidth = 2;
  const failed = map.cells.map((row) => row.map((c) => c.failed));
  for (const e of boundaryEdges(failed)) {
    ctx.beginPath();
    if (e.side === "left") { ctx.moveTo(X(e.i), Y(e.j)); ctx.lineTo(X(e.i), Y(e.j + 1)); }
    else { ctx.moveTo(X(e.i), Y(e.j)); ctx.lineTo(X(e.i + 1), Y(e.j)); }
    ctx.stroke();
  }

  const cx = X(current.i) + cw / 2;
  const cy = Y(current.j) + ch / 2;
  ctx.strokeStyle = token("--text-0");
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy); ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10); ctx.stroke();
  ctx.strokeRect(X(current.i), Y(current.j), cw, ch);

  ctx.fillStyle = token("--text-1");
  ctx.font = `11px ${token("--font-mono")}, monospace`;
  const ticks = (n: number) => [0, Math.round((n - 1) / 4), Math.round((n - 1) / 2), Math.round(3 * (n - 1) / 4), n - 1];
  ctx.textAlign = "center";
  for (const i of ticks(nx)) ctx.fillText(fmt(map.x.values[i], map.x.unit, 3), X(i) + cw / 2, Y(ny) + 16);
  ctx.fillText(`exposure (${map.x.unit}) — blur and light from one shutter`, PAD.left + (nx * cw) / 2, Y(ny) + 36);
  ctx.textAlign = "right";
  for (const j of ticks(ny)) ctx.fillText(fmt(map.y.values[j], map.y.unit, 3), PAD.left - 6, Y(j) + ch / 2 + 4);
  ctx.textAlign = "left";
}

export default function Hero({ report }: { report: Report }) {
  const map = report.map2d!;
  const stops = useMemo(() => imageStops(map.cells), [map]);
  const uncovered = useMemo(() => uncoveredMask(map, report.uncovered_regions), [map, report]);
  // ?x=&y= select slider stops, so a given view can be linked and reproduced.
  const initial = (key: string, n: number) => {
    if (typeof window === "undefined") return 0;
    const v = Number(new URLSearchParams(window.location.search).get(key));
    return Number.isInteger(v) && v >= 0 && v < n ? v : 0;
  };
  const [xi, setXi] = useState(() => initial("x", stops.xs.length));
  const [yi, setYi] = useState(() => initial("y", stops.ys.length));
  const cell = map.cells[stops.ys[yi]][stops.xs[xi]];
  const images = useImages(map, cell.i, cell.j);
  const frameRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<HTMLCanvasElement>(null);
  const baseline = report.run.baseline_map50;

  useEffect(() => {
    const f = frameRef.current?.getContext("2d");
    const m = mapRef.current?.getContext("2d");
    if (f) drawFrame(f, cell.image ? images[cell.image] : undefined, cell);
    if (m) drawMap(m, map, baseline, uncovered, cell);
  }, [cell, images, map, baseline, uncovered]);

  const wrong = (cell.detections?.predictions ?? []).filter((p) => !p.matched)
    .sort((a, b) => b.score - a.score)[0];
  const command = reproduceCommand(cell.exposure_ms, cell.illuminance_lux, map.seed);

  return (
    <section aria-label="Blindspot Map">
      <div className="hero">
        <div>
          <canvas ref={frameRef} width={FRAME_W} height={FRAME_H} className="viewport"
                  aria-label="Measured frame under the selected condition" />
          <div className="caption">
            frame {map.demo_frame.image_id} ({map.demo_frame.attribution.license},{" "}
            <a href={map.demo_frame.attribution.source}>source</a>) · faces blurred · boxes ≥ 0.50 shown ·
            solid = prediction, dashed = ground truth
          </div>
        </div>
        <div>
          <canvas ref={mapRef} width={MAP_W} height={MAP_H} className="heatmap"
                  aria-label="Blindspot Map: mAP@50 over exposure and illuminance" />
          <div className="legend">
            <span><i className="swatch" style={{ background: viridis(1) }} />passes (mAP / baseline)</span>
            <span><i className="swatch" style={{ background: "var(--fail-dim)" }} />fails, hatched</span>
            <span><i className="swatch" style={{ background: "var(--uncovered)" }} />dots: not covered by your data</span>
            <span><i className="swatch" style={{ background: "var(--fail)" }} />boundary</span>
          </div>
          <div className="caption">
            {map.n_failed} of {map.n_cells} measured conditions fail · {map.coupling} ·
            angular velocity {String(map.camera_model.angular_velocity_deg_s)} deg/s · fixed gain
          </div>
        </div>
      </div>

      <div className="sliders">
        <label className="slider">
          <span className="label">exposure</span>
          <input type="range" min={0} max={stops.xs.length - 1} step={1} value={xi}
                 onChange={(e) => setXi(+e.target.value)} aria-valuetext={fmt(cell.exposure_ms, "ms")} />
          <span className="value">{fmt(cell.exposure_ms, "ms")} · {fmt(cell.psf_length_px, "px", 3)}</span>
        </label>
        <label className="slider">
          <span className="label">illuminance</span>
          <input type="range" min={0} max={stops.ys.length - 1} step={1} value={yi}
                 onChange={(e) => setYi(+e.target.value)} aria-valuetext={fmt(cell.illuminance_lux, "lux")} />
          <span className="value">{fmt(cell.illuminance_lux, "lux", 3)}</span>
        </label>
      </div>

      <div className="measure-bar">
        <div className={`verdict ${cell.failed ? "fail" : "pass"}`}>
          {cell.failed ? "FAIL BOUNDARY CROSSED" : "within envelope"} ·{" "}
          {fmt(cell.exposure_ms, "ms")} @ {fmt(cell.illuminance_lux, "lux", 3)}
          {wrong && <> · most confident wrong box: {wrong.label} {wrong.score.toFixed(2)}</>}
        </div>
        <div className="verdict mono">
          mAP@50 {baseline.toFixed(3)} → {cell.map50.toFixed(3)}
          <span className="caption"> (fail below {report.run.threshold_map50.toFixed(3)})</span>
        </div>
        <button className="reproduce" title="copy" onClick={() => navigator.clipboard?.writeText(command)}>
          reproduce: {command}
        </button>
      </div>
    </section>
  );
}
