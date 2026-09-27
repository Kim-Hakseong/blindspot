"""2-D probe savings: level-set estimation against the exhaustive map (W3-3).

The exhaustive exposure x illuminance map (`bench/out/hook_grid.json`) probed
every cell on the full validation set. Because a probe is a deterministic
function of its condition and seed, looking a cell up in that map returns
exactly what running the probe again would -- so the map is an exact oracle,
and every "probe" the sampler makes here is counted as one.

Savings are only reported as valid under the same rule as the 1-D benchmark:
every cell the sampler classifies differently from the exhaustive map must lie
within one cell of the true boundary. A sampler that saves probes by getting
the boundary wrong has saved nothing.

    uv run python bench/levelset_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from blindspot.boundary.levelset import estimate_level_set  # noqa: E402


def chebyshev_to_boundary(truth: np.ndarray, j: int, i: int) -> int:
    opposite = np.argwhere(truth != truth[j, i])
    if not len(opposite):
        return 10**6
    return int(np.min(np.max(np.abs(opposite - [j, i]), axis=1)))


def evaluate_map(grid: dict, beta: float) -> dict:
    values = np.array([[c["map50"] for c in row] for row in grid["cells"]])
    truth = np.array([[c["failed"] for c in row] for row in grid["cells"]])
    threshold = grid["criterion"]["threshold_map50"]
    ny, nx = values.shape

    order: list[tuple[int, int]] = []

    def evaluate(i: int, j: int) -> float:
        order.append((i, j))
        return float(values[j, i])

    result = estimate_level_set(evaluate, nx, ny, threshold, beta=beta)
    wrong = np.argwhere(result.failed != truth)
    distances = [chebyshev_to_boundary(truth, int(j), int(i)) for j, i in wrong]
    valid = all(d <= 1 for d in distances)
    return {
        "beta": beta,
        "grid_cells": int(nx * ny),
        "probes_used": result.probes_used,
        "status": result.status,
        "misclassified_cells": int(len(wrong)),
        "max_distance_to_boundary_cells": max(distances) if distances else 0,
        "savings_valid": bool(valid),
        "savings": round(nx * ny / result.probes_used, 4) if valid else None,
        "probe_order": order,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=pathlib.Path, default=pathlib.Path("bench/out/hook_grid.json"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/levelset_efficiency.json"))
    parser.add_argument("--betas", type=float, nargs="*", default=[1.0, 1.96, 3.0])
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    runs = [evaluate_map(grid, b) for b in args.betas]
    primary = next(r for r in runs if r["beta"] == 1.96)
    report = {
        "oracle": "exhaustive map bench/out/hook_grid.json; deterministic probes make lookup equal to re-running",
        "axes": [grid["x"]["axis"], grid["y"]["axis"]],
        "coupling": grid["coupling"],
        "grid_steps": grid["steps"],
        "dataset": grid["dataset"],
        "pipeline": grid["pipeline"]["name"],
        "threshold_map50": grid["criterion"]["threshold_map50"],
        "validity_rule": "every misclassified cell within 1 cell (Chebyshev) of the true boundary",
        "primary_beta": 1.96,
        "primary": {k: v for k, v in primary.items() if k != "probe_order"},
        "sensitivity": [{k: v for k, v in r.items() if k != "probe_order"} for r in runs],
        "probe_order_primary": primary["probe_order"],
        "command": "uv run python bench/levelset_efficiency.py",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in runs:
        s = f"{r['savings']:.2f}x" if r["savings"] else "invalid"
        print(f"beta={r['beta']:<5} probes {r['probes_used']:>3}/{r['grid_cells']}  "
              f"misclassified {r['misclassified_cells']} (max dist {r['max_distance_to_boundary_cells']})  "
              f"savings {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
