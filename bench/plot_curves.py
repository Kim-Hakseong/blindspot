"""Condition curves: mAP@50 against each physical axis.

Reads the committed grid sweeps and draws what they measured -- nothing is
re-derived here. The threshold is the criterion's, and the boundary band is the
interval the grid reported, drawn as a band rather than a line because the
boundary is an interval.

Colours follow the report viewer's tokens so a curve and the viewer read as the
same instrument. The failing region carries a hatch as well as a colour, so it
does not depend on colour vision.

    uv run python bench/plot_curves.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOKENS = {
    "bg0": "#0B0E14",
    "bg1": "#12161F",
    "border": "#252B38",
    "text0": "#E8ECF2",
    "text1": "#A3AEBF",
    "text2": "#6B7684",
    "fail": "#FF4D4D",
    "fail_dim": "#7A1F26",
    "accent": "#4C8DFF",
}


@dataclass(frozen=True)
class Curve:
    axis: str
    unit: str
    values: list[float]
    map50: list[float]
    threshold: float
    baseline: float
    boundary: tuple[float, float] | None
    command: str

    @property
    def x_label(self) -> str:
        return f"{self.axis}  [{self.unit}]"


def curve_from_grid(grid: dict) -> Curve:
    probes = sorted(grid["probes"], key=lambda p: p["value"])
    boundary = grid.get("boundary")
    return Curve(
        axis=grid["axis"],
        unit=grid["unit"],
        values=[float(p["value"]) for p in probes],
        map50=[float(p["map50"]) for p in probes],
        threshold=float(grid["criterion"]["threshold_map50"]),
        baseline=float(grid["criterion"]["baseline_map50"]),
        boundary=(float(boundary["lower"]), float(boundary["upper"])) if boundary else None,
        command=grid.get("command", ""),
    )


def render(curve: Curve, out: pathlib.Path) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor(TOKENS["bg0"])
    ax.set_facecolor(TOKENS["bg1"])
    for spine in ax.spines.values():
        spine.set_color(TOKENS["border"])
    ax.tick_params(colors=TOKENS["text1"], labelsize=9)
    ax.grid(color=TOKENS["border"], linewidth=0.5)

    ax.axhspan(0.0, curve.threshold, facecolor=TOKENS["fail_dim"], alpha=0.35,
               hatch="///", edgecolor=TOKENS["fail_dim"], linewidth=0)
    ax.axhline(curve.threshold, color=TOKENS["fail"], linewidth=1.2)
    ax.text(curve.values[-1], curve.threshold, f" fail < {curve.threshold:.3f}",
            color=TOKENS["fail"], fontsize=8, family="monospace", va="bottom", ha="right")

    if curve.boundary:
        lo, hi = curve.boundary
        ax.axvspan(lo, hi, color=TOKENS["fail"], alpha=0.2, linewidth=0)
        ax.axvline(lo, color=TOKENS["fail"], linewidth=2)
        ax.text(hi, 0.02, f" boundary {lo:.4g}-{hi:.4g} {curve.unit}",
                color=TOKENS["fail"], fontsize=8, family="monospace")

    ax.plot(curve.values, curve.map50, color=TOKENS["accent"], linewidth=1.6,
            marker="o", markersize=3)
    ax.set_ylim(0.0, max(max(curve.map50), curve.baseline) * 1.1)
    ax.set_xlabel(curve.x_label, color=TOKENS["text1"], family="monospace", fontsize=9)
    ax.set_ylabel("mAP@50", color=TOKENS["text1"], family="monospace", fontsize=9)
    ax.set_title(curve.axis, color=TOKENS["text0"], fontsize=11, loc="left")
    fig.text(0.01, 0.01, curve.command, color=TOKENS["text2"], fontsize=6, family="monospace")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def render_all(grids: list[pathlib.Path], out_dir: pathlib.Path) -> list[pathlib.Path]:
    written, index = [], []
    for path in grids:
        curve = curve_from_grid(json.loads(path.read_text(encoding="utf-8")))
        png = render(curve, out_dir / f"{path.stem}.png")
        written.append(png)
        index.append({"source": path.name, "png": png.name, "axis": curve.axis,
                      "unit": curve.unit, "boundary": curve.boundary})
    (out_dir / "curves.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="*", type=pathlib.Path,
                        default=sorted(pathlib.Path("bench/out").glob("grid_*.json")))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/out/curves"))
    args = parser.parse_args()
    for png in render_all(args.grids, args.out):
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
