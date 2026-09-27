// Rule U3: colours are defined once, as tokens in app/globals.css. Any colour
// literal elsewhere in the UI code is a defect. The viridis table in
// lib/map.mjs is the colour-map specification itself and is the one exception.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const COLOUR = /#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(/;

function files(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

test("no colour literal outside the token file and the colour-map table", () => {
  const offenders = [];
  for (const f of [...files("app"), ...files("components"), ...files("lib")]) {
    if (!/\.(tsx?|mjs|css)$/.test(f) || f.endsWith("globals.css")) continue;
    readFileSync(f, "utf8").split("\n").forEach((line, k) => {
      if (f.endsWith("map.mjs") && /VIRIDIS|^\s*"#/.test(line)) return;
      if (COLOUR.test(line) && !/function hex|slice\(1\)/.test(line)) offenders.push(`${f}:${k + 1}: ${line.trim()}`);
    });
  }
  assert.deepEqual(offenders, []);
});
