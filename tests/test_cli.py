"""CLI reproduction commands (rule R2).

Every command the report prints must actually run and must reproduce the
number it sits next to. These run on two frames so they finish in seconds.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from blindspot.cli import app

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_assets = pytest.mark.skipif(
    not (ROOT / "models" / "yolox_s.onnx").is_file()
    or not (ROOT / "val" / "road100" / "manifest.json").is_file(),
    reason="model or dataset not fetched",
)
runner = CliRunner()


def run(*args):
    result = runner.invoke(app, ["probe", *args, "--frames", "2"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{"):])


@needs_assets
def test_composite_probe_applies_set_and_fix():
    out = run("--set", "motion_blur.exposure_ms=12.375",
              "--set", "low_light.illuminance_lux=18.3",
              "--fix", "low_light.exposure_ms=12.375", "--seed", "20260906")
    assert {"axis": "low_light.exposure_ms", "value": 12.375, "unit": "ms", "fixed": True} in out["condition"]
    assert 0.0 <= out["map50"] <= 1.0


@needs_assets
def test_composite_probe_is_reproducible():
    args = ("--set", "fog.beta_per_m=0.05", "--seed", "3")
    assert run(*args)["map50"] == run(*args)["map50"]


@needs_assets
def test_single_axis_probe_still_works():
    out = run("--degradation", "jpeg", "--axis", "quality", "--value", "10", "--seed", "1")
    assert out["degradation"] == "jpeg"


def test_probe_needs_either_set_or_a_single_axis():
    result = runner.invoke(app, ["probe", "--frames", "1"])
    assert result.exit_code != 0


def test_malformed_set_is_rejected():
    result = runner.invoke(app, ["probe", "--set", "motion_blur.exposure_ms", "--frames", "1"])
    assert result.exit_code != 0


@needs_assets
def test_the_command_a_probe_prints_actually_runs_and_reproduces_it():
    """R2: the reproduce command stored with a result must parse and give the
    same number. Found broken: it once printed every parameter as a flag the
    CLI does not accept."""
    import shlex

    first = run("--degradation", "motion_blur", "--axis", "exposure_ms", "--value", "12.5", "--seed", "4")
    argv = shlex.split(first["command"])
    assert argv[:3] == ["uv", "run", "blindspot"]
    again = run(*argv[4:])  # drop "uv run blindspot probe"
    assert again["map50"] == first["map50"]
