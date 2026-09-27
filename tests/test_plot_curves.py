"""Condition curves (W2-6).

The plot is evidence, so the data behind it is tested rather than eyeballed:
the curve must be ordered along the physical axis, the threshold must be the
one the criterion defined, and the boundary band must be exactly the interval
the grid reported -- not something the plotting code re-derived.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_curves", ROOT / "bench" / "plot_curves.py")
plot_curves = importlib.util.module_from_spec(SPEC)
sys.modules["plot_curves"] = plot_curves
SPEC.loader.exec_module(plot_curves)


def fake_grid(values, maps, threshold=0.36, boundary=(10.0, 12.0), unit="ms"):
    return {
        "axis": "motion_blur.exposure_ms",
        "unit": unit,
        "criterion": {"threshold_map50": threshold, "baseline_map50": 0.6},
        "boundary": {"lower": boundary[0], "upper": boundary[1], "unit": unit}
        if boundary
        else None,
        "command": "uv run python bench/grid_baseline.py --axis motion_blur.exposure_ms",
        "probes": [{"value": v, "map50": m} for v, m in zip(values, maps)],
    }


def test_curve_is_sorted_by_axis_value_even_if_probes_are_not():
    grid = fake_grid([20.0, 0.0, 10.0], [0.1, 0.6, 0.4])
    curve = plot_curves.curve_from_grid(grid)
    assert curve.values == [0.0, 10.0, 20.0]
    assert curve.map50 == [0.6, 0.4, 0.1]


def test_curve_carries_the_criterion_threshold_unchanged():
    curve = plot_curves.curve_from_grid(fake_grid([0, 1], [0.6, 0.1], threshold=0.3666))
    assert curve.threshold == pytest.approx(0.3666)


def test_curve_boundary_is_the_grid_reported_interval():
    curve = plot_curves.curve_from_grid(fake_grid([0, 1], [0.6, 0.1], boundary=(12.5, 13.75)))
    assert curve.boundary == (12.5, 13.75)


def test_curve_without_a_boundary_reports_none():
    curve = plot_curves.curve_from_grid(fake_grid([0, 1], [0.6, 0.5], boundary=None))
    assert curve.boundary is None


def test_curve_label_includes_the_physical_unit():
    curve = plot_curves.curve_from_grid(fake_grid([0, 1], [0.6, 0.1], unit="lux"))
    assert "lux" in curve.x_label


def test_render_writes_a_png(tmp_path):
    curve = plot_curves.curve_from_grid(fake_grid([0, 5, 10, 15], [0.6, 0.5, 0.3, 0.1]))
    out = tmp_path / "curve.png"
    plot_curves.render(curve, out)
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 2000


def test_render_all_on_committed_grids(tmp_path):
    """The committed benchmark outputs must all be plottable."""
    grids = sorted((ROOT / "bench" / "out").glob("grid_*.json"))
    if not grids:
        pytest.skip("no committed grid outputs")
    written = plot_curves.render_all(grids, tmp_path)
    assert len(written) == len(grids)
    for path in written:
        assert path.stat().st_size > 2000
        assert json.loads((tmp_path / "curves.json").read_text())
