// Pure map logic behind the Hero screen. The heatmap is evidence, so the
// boundary it draws, the cells a slider can reach and the "uncovered" marking
// are tested here rather than eyeballed in a browser.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  boundaryEdges, imageStops, uncoveredMask, viridis, reproduceCommand, fmt,
} from "../lib/map.mjs";

const F = true, P = false;

test("boundary edges lie exactly between pass and fail neighbours", () => {
  const failed = [
    [P, F],
    [P, F],
  ];
  const edges = boundaryEdges(failed);
  // One vertical edge between column 0 and 1, on both rows.
  assert.deepEqual(edges, [
    { i: 1, j: 0, side: "left" },
    { i: 1, j: 1, side: "left" },
  ]);
});

test("no edges when everything fails or everything passes", () => {
  assert.deepEqual(boundaryEdges([[F, F], [F, F]]), []);
  assert.deepEqual(boundaryEdges([[P, P], [P, P]]), []);
});

test("horizontal edges are found too", () => {
  const edges = boundaryEdges([[P], [F]]);
  assert.deepEqual(edges, [{ i: 0, j: 1, side: "top" }]);
});

test("slider stops are only the cells that have a scored frame image", () => {
  const cells = [
    [{ i: 0, j: 0, image: "a" }, { i: 1, j: 0, image: null }, { i: 2, j: 0, image: "b" }],
    [{ i: 0, j: 1, image: null }, { i: 1, j: 1, image: null }, { i: 2, j: 1, image: null }],
    [{ i: 0, j: 2, image: "c" }, { i: 1, j: 2, image: null }, { i: 2, j: 2, image: "d" }],
  ];
  assert.deepEqual(imageStops(cells), { xs: [0, 2], ys: [0, 2] });
});

test("a cell is uncovered if either coordinate lies in an uncovered interval", () => {
  const map = {
    x: { axis: "motion_blur.exposure_ms", values: [1, 10, 20] },
    y: { axis: "low_light.illuminance_lux", values: [400, 50, 1] },
  };
  const regions = [
    { axis: "motion_blur.exposure_ms", uncovered: [{ from: 15, to: 40 }] },
    { axis: "low_light.illuminance_lux", uncovered: [{ from: 0.5, to: 5 }] },
  ];
  assert.deepEqual(uncoveredMask(map, regions), [
    [false, false, true],
    [false, false, true],
    [true, true, true],
  ]);
});

test("uncovered mask is all false when the axes have no coverage record", () => {
  const map = { x: { axis: "a", values: [1, 2] }, y: { axis: "b", values: [1] } };
  assert.deepEqual(uncoveredMask(map, []), [[false, false]]);
});

test("viridis endpoints and clamping", () => {
  assert.equal(viridis(0), "#440154");
  assert.equal(viridis(1), "#fde725");
  assert.equal(viridis(-3), viridis(0));
  assert.equal(viridis(9), viridis(1));
});

test("reproduce command pins the shared exposure and the seed", () => {
  const cmd = reproduceCommand(12.375, 18.3, 20260906);
  assert.match(cmd, /--set motion_blur\.exposure_ms=12\.375/);
  assert.match(cmd, /--set low_light\.illuminance_lux=18\.3/);
  assert.match(cmd, /--fix low_light\.exposure_ms=12\.375/);
  assert.match(cmd, /--seed 20260906/);
});

test("every formatted number carries its unit", () => {
  assert.equal(fmt(12.3456, "ms"), "12.35 ms");
  assert.equal(fmt(0.5, "lux"), "0.5 lux");
  assert.throws(() => fmt(1, ""));
});

test("reproduce command carries the exact scored value, not a rounded one", () => {
  const lux = 173.20508075688772;
  const cmd = reproduceCommand(3.4375, lux, 1);
  const printed = Number(cmd.match(/illuminance_lux=([^ ]+)/)[1]);
  assert.equal(printed, lux);
});
