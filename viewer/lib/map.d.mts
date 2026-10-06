export type Edge = { i: number; j: number; side: "left" | "top" };
export function boundaryEdges(failed: boolean[][]): Edge[];
export function imageStops(cells: { i: number; j: number; image: string | null }[][]): { xs: number[]; ys: number[] };
export function uncoveredMask(
  map: { x: { axis: string; values: number[] }; y: { axis: string; values: number[] } },
  regions: { axis: string; uncovered: { from: number; to: number }[] }[],
): boolean[][];
export function viridis(t: number): string;
export function reproduceCommand(exposureMs: number, lux: number, seed: number): string;
export function fmt(value: number, unit: string, digits?: number): string;
export function loadOrder(cells: { i: number; j: number; image: string | null }[][], i: number, j: number): string[];
