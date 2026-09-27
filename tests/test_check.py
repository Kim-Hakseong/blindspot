"""The CI gate: fail when a boundary regresses (blindspot check).

A regression means the pipeline now fails under milder conditions. For axes
whose severe end is high (exposure, fog) that is the boundary moving down; for
axes whose severe end is low (illuminance, JPEG quality) it is the boundary
moving *up*.
"""

import json

from typer.testing import CliRunner

from blindspot.cli import app


def report(tmp_path, name, **bounds):
    findings = [{"axis": a, "unit": "u", "status": "located",
                 "boundary": {"lower": lo, "upper": hi, "unit": "u"}} for a, (lo, hi) in bounds.items()]
    p = tmp_path / name
    p.write_text(json.dumps({"findings": findings}))
    return p


def run(old, new):
    return CliRunner().invoke(app, ["check", "--against", str(old), "--current", str(new)])


def test_exposure_boundary_moving_down_is_a_regression(tmp_path):
    old = report(tmp_path, "a.json", **{"motion_blur.exposure_ms": (12.5, 13.75)})
    new = report(tmp_path, "b.json", **{"motion_blur.exposure_ms": (10.0, 11.25)})
    assert run(old, new).exit_code == 1


def test_exposure_boundary_moving_up_is_an_improvement(tmp_path):
    old = report(tmp_path, "a.json", **{"motion_blur.exposure_ms": (12.5, 13.75)})
    new = report(tmp_path, "b.json", **{"motion_blur.exposure_ms": (15.0, 16.25)})
    assert run(old, new).exit_code == 0


def test_illuminance_boundary_moving_up_is_a_regression(tmp_path):
    """Now failing in brighter light: worse, although the number went up."""
    old = report(tmp_path, "a.json", **{"low_light.illuminance_lux": (12.98, 25.47)})
    new = report(tmp_path, "b.json", **{"low_light.illuminance_lux": (25.47, 37.95)})
    result = run(old, new)
    assert result.exit_code == 1, result.output
    assert "low_light.illuminance_lux" in result.output


def test_illuminance_boundary_moving_down_is_an_improvement(tmp_path):
    old = report(tmp_path, "a.json", **{"low_light.illuminance_lux": (12.98, 25.47)})
    new = report(tmp_path, "b.json", **{"low_light.illuminance_lux": (0.5, 12.98)})
    assert run(old, new).exit_code == 0


def test_a_boundary_that_appears_where_none_was_is_a_regression(tmp_path):
    old = tmp_path / "a.json"
    old.write_text(json.dumps({"findings": [{"axis": "fog.beta_per_m", "status": "passes_throughout",
                                             "boundary": None}]}))
    new = report(tmp_path, "b.json", **{"fog.beta_per_m": (0.06, 0.064)})
    assert run(old, new).exit_code == 1
