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


def _report(label, totals_ms, maps):
    return {"label": label, "fingerprint": {"cpu_model": label, "machine": "m", "kleidicv": False,
                                            "opencv_version": "5.0.0"},
            "per_frame": {s: {"median_ms": t, "p95_ms": t, "total_s": 1.0, "n": 1}
                          for s, t in zip(("degrade", "measure", "infer"), totals_ms)},
            "per_frame_total_median_ms": sum(totals_ms),
            "results": [{"probe_id": i, "map50": m} for i, m in enumerate(maps)]}


def test_compare_arms_prices_a_frame_and_checks_agreement():
    from blindspot.stages import compare_arms
    arms = {"arm64": [_report("a", (1, 1, 8), [0.5, 0.25]), _report("a", (1, 1, 10), [0.5, 0.25])],
            "x86": [_report("b", (2, 2, 16), [0.5, 0.2501])]}
    prices = {"arm64": 0.036, "x86": 0.072}  # USD per task-hour
    out = compare_arms(arms, prices)
    # median over repeats of the per-frame total median
    assert out["arms"]["arm64"]["per_frame_total_median_ms"] == 11.0
    assert out["arms"]["x86"]["per_frame_total_median_ms"] == 20.0
    # cost of 1000 frames = 1000 * ms/3.6e6 h * $/h
    assert out["arms"]["arm64"]["usd_per_1000_frames"] == pytest.approx(1000 * 11 / 3.6e6 * 0.036)
    assert out["speedup_arm64_over_x86"] == pytest.approx(20 / 11)
    assert out["map50_identical_probes"] == 1 and out["map50_max_abs_difference"] == pytest.approx(1e-4)
    # repeats on one arm must agree exactly, or the arm is not deterministic
    assert out["arms"]["arm64"]["repeats_bit_identical"] is True


def test_compare_arms_exposes_mixed_cpus_within_an_arm():
    # Fargate does not pin a CPU generation: one x86 arm can land on several.
    from blindspot.stages import compare_arms
    arms = {"arm64": [_report("v2", (1, 1, 8), [0.5])],
            "x86": [_report("old", (2, 2, 20), [0.5]), _report("new", (1, 1, 10), [0.5])]}
    out = compare_arms(arms, {"arm64": 1.0, "x86": 1.0})
    assert out["arms"]["x86"]["cpu_models"] == {"old": 1, "new": 1}
    assert out["arms"]["x86"]["runs"] == [{"cpu_model": "old", "per_frame_total_median_ms": 24},
                                          {"cpu_model": "new", "per_frame_total_median_ms": 12}]
    assert out["arms"]["arm64"]["cpu_models"] == {"v2": 1}
