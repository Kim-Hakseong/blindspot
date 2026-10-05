"""Per-stage timing for the CPU comparison (x86 / Graviton / COOL).

Timing is measured, so these tests check structure and correctness rather
than speed: the fixed probe batch, the per-stage breakdown, the CPU
fingerprint, and that mAP is reported per probe so arms can be compared for
agreement as well as speed.
"""

import pathlib

import pytest

from blindspot.stages import fixture_probes, fingerprint, run_stage_bench

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_assets = pytest.mark.skipif(
    not (ROOT / "models" / "yolox_s.onnx").is_file()
    or not (ROOT / "val" / "road100" / "manifest.json").is_file(),
    reason="model or dataset not fetched")


def test_fixture_is_64_probes_16_per_axis_and_deterministic():
    a, b = fixture_probes(), fixture_probes()
    assert a == b and len(a) == 64
    axes = [next(iter(p["set"])) for p in a]
    assert {axes.count(x) for x in set(axes)} == {16}


def test_fingerprint_names_the_cpu_and_the_opencv_build():
    fp = fingerprint()
    assert {"machine", "cpu_model", "opencv_version", "build_info_sha256", "kleidicv"} <= set(fp)


@needs_assets
def test_stage_bench_reports_each_stage_and_per_probe_map(tmp_path):
    report = run_stage_bench(ROOT / "val" / "road100", frames=2, probes=fixture_probes()[:3],
                             label="test")
    assert set(report["per_frame"]) == {"degrade", "measure", "infer"}
    assert len(report["results"]) == 3
    assert all(0.0 <= r["map50"] <= 1.0 for r in report["results"])
