// Pure logic behind the Blindspot Map. No DOM, no React -- tested with
// `node --test`. Everything the Hero screen draws as evidence is computed here.

/** Segments between a passing and a failing cell. `side` is the side of cell
 * (i, j) the edge lies on, so the renderer draws it without re-deriving. */
export function boundaryEdges(failed) {
  const edges = [];
  for (let j = 0; j < failed.length; j++) {
    for (let i = 0; i < failed[j].length; i++) {
      if (i > 0 && failed[j][i] !== failed[j][i - 1]) edges.push({ i, j, side: "left" });
    }
  }
  for (let j = 1; j < failed.length; j++) {
    for (let i = 0; i < failed[j].length; i++) {
      if (failed[j][i] !== failed[j - 1][i]) edges.push({ i, j, side: "top" });
    }
  }
  return edges;
}

/** Column and row indices that have a scored frame image. Sliders snap to
 * these: the viewer never interpolates a frame it did not measure. */
export function imageStops(cells) {
  const xs = new Set();
  const ys = new Set();
  for (const row of cells) for (const c of row) if (c.image) { xs.add(c.i); ys.add(c.j); }
  const sort = (s) => [...s].sort((a, b) => a - b);
  return { xs: sort(xs), ys: sort(ys) };
}

function inAny(value, intervals) {
  return intervals.some((r) => value >= Math.min(r.from, r.to) && value <= Math.max(r.from, r.to));
}

/** True where a cell's condition lies outside what the validation set covers
 * on either axis -- conditions nobody has natural evidence about. */
export function uncoveredMask(map, regions) {
  const byAxis = Object.fromEntries(regions.map((r) => [r.axis, r.uncovered]));
  const xu = byAxis[map.x.axis] ?? [];
  const yu = byAxis[map.y.axis] ?? [];
  return map.y.values.map((y) => map.x.values.map((x) => inAny(x, xu) || inAny(y, yu)));
}

// matplotlib viridis, sampled at 9 stops (colour-blind-safe sequential map).
const VIRIDIS = [
  "#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
  "#28ae80", "#5ec962", "#addc30", "#fde725",
];

function hex(c) { return parseInt(c.slice(1), 16); }

export function viridis(t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const pos = x * (VIRIDIS.length - 1);
  const k = Math.min(VIRIDIS.length - 2, Math.floor(pos));
  const f = pos - k;
  if (f === 0) return VIRIDIS[k];
  if (x === 1) return VIRIDIS[VIRIDIS.length - 1];
  const a = hex(VIRIDIS[k]);
  const b = hex(VIRIDIS[k + 1]);
  const mix = (sh) => Math.round(((a >> sh) & 255) * (1 - f) + ((b >> sh) & 255) * f);
  return "#" + [16, 8, 0].map((sh) => mix(sh).toString(16).padStart(2, "0")).join("");
}

/** The command that re-runs exactly the cell on screen. */
export function reproduceCommand(exposureMs, lux, seed) {
  // String(n) is the shortest round-trip form: rounding here would print a
  // *different* condition from the one that was scored.
  const e = String(exposureMs);
  const l = String(lux);
  return (
    `uv run blindspot probe --set motion_blur.exposure_ms=${e} ` +
    `--set low_light.illuminance_lux=${l} --fix low_light.exposure_ms=${e} --seed ${seed}`
  );
}

/** A number is only ever shown with its unit. */
export function fmt(value, unit, digits = 4) {
  if (!unit) throw new Error("a number without a unit is not displayed");
  return `${+Number(value).toPrecision(digits)} ${unit}`;
}

/** Cell images in the order to fetch them: the cell on screen first, then
 * outward by grid distance, so the frame a user is looking at never waits
 * behind 80 others. */
export function loadOrder(cells, i, j) {
  const seen = new Set();
  return cells.flat()
    .filter((c) => c.image)
    .sort((a, b) => Math.hypot(a.i - i, a.j - j) - Math.hypot(b.i - i, b.j - j))
    .map((c) => c.image)
    .filter((name) => !seen.has(name) && seen.add(name));
}
