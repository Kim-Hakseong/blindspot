"""The report the viewer reads (W5-4).

The schema encodes three rules as types rather than as good intentions:
uncovered regions come first (R1), every boundary carries the recipe that
regenerates it (R2), and a quantity that was not measured is the string
"not measured" -- never 0, never null that a chart might draw as 0.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from blindspot.report import NOT_MEASURED, Report


def minimal(**overrides):
    doc = {
        "uncovered_regions": [
            {"axis": "fog.beta_per_m", "unit": "1/m", "coverage_percent": 20.0,
             "uncovered": [{"from": 0.03, "to": 0.12, "unit": "1/m"}]}
        ],
        "run": {"run_id": "local-1", "pipeline": "yolox_s", "dataset": "road100",
                "frames": 100, "objects": 984, "seed": 1,
                "baseline_map50": 0.61, "threshold_map50": 0.366,
                "criterion": "mAP@50 below 60% of baseline"},
        "findings": [
            {"axis": "motion_blur.exposure_ms", "unit": "ms", "status": "located",
             "lower": 12.5, "upper": 13.75,
             "reproduce": {"axis": "motion_blur.exposure_ms", "value": 13.75, "unit": "ms",
                           "seed": 1, "source_frame": "210273",
                           "command": "uv run blindspot probe --degradation motion_blur "
                                      "--axis exposure_ms --value 13.75 --seed 1"}}
        ],
        "map2d": None,
        "curves": [],
        "efficiency": None,
        "evidence_frames": [],
        "measurements": {"sim2real_gap": NOT_MEASURED, "cool_vs_stock": NOT_MEASURED, "graviton_vs_x86": NOT_MEASURED},
        "limitations": ["a", "b", "c", "d", "e"],
    }
    doc.update(overrides)
    return doc


def test_minimal_report_validates():
    Report.model_validate(minimal())


def test_uncovered_regions_is_the_first_key_when_serialised():
    text = Report.model_validate(minimal()).dump_json()
    assert next(iter(json.loads(text))) == "uncovered_regions"


def test_a_finding_without_a_reproduction_recipe_is_rejected():
    doc = minimal()
    del doc["findings"][0]["reproduce"]
    with pytest.raises(ValidationError):
        Report.model_validate(doc)


def test_reproduction_recipe_requires_every_field():
    for field in ("axis", "value", "unit", "seed", "source_frame", "command"):
        doc = minimal()
        del doc["findings"][0]["reproduce"][field]
        with pytest.raises(ValidationError):
            Report.model_validate(doc)


def test_a_located_finding_needs_both_ends_of_its_interval():
    doc = minimal()
    doc["findings"][0]["upper"] = None
    with pytest.raises(ValidationError):
        Report.model_validate(doc)


def test_interval_must_be_ordered():
    doc = minimal()
    doc["findings"][0]["lower"], doc["findings"][0]["upper"] = 13.75, 12.5
    with pytest.raises(ValidationError):
        Report.model_validate(doc)


@pytest.mark.parametrize("bad", [0, 0.0, None, "", "n/a", "-"])
def test_an_unmeasured_quantity_cannot_be_zero_or_blank(bad):
    doc = minimal(measurements={"sim2real_gap": bad, "cool_vs_stock": NOT_MEASURED, "graviton_vs_x86": NOT_MEASURED})
    with pytest.raises(ValidationError):
        Report.model_validate(doc)


def test_a_measured_quantity_carries_value_unit_and_source():
    doc = minimal(measurements={
        "sim2real_gap": {"value": 3.2, "unit": "ms", "source": "bench/out/sim2real.json"},
        "cool_vs_stock": NOT_MEASURED, "graviton_vs_x86": NOT_MEASURED,
    })
    Report.model_validate(doc)


def test_fewer_than_five_limitations_is_rejected():
    with pytest.raises(ValidationError):
        Report.model_validate(minimal(limitations=["only", "four", "known", "limits"]))


def test_unknown_top_level_fields_are_rejected():
    with pytest.raises(ValidationError):
        Report.model_validate(minimal(passed_conditions=[]))


def test_measurements_are_read_from_the_benchmark_outputs(tmp_path):
    # The viewer's "measured" items come from bench/out, never typed in.
    import json
    from blindspot.report import measurements_from_bench
    (tmp_path / "cool" / "m8g-4xlarge").mkdir(parents=True)
    (tmp_path / "sim2real.json").write_text(json.dumps({"gap": {
        "synthetic_boundary_overstates_failure_illuminance_by_at_least": 12.0}}))
    (tmp_path / "cool" / "m8g-4xlarge" / "ec2_three_way.json").write_text(json.dumps(
        {"cool_effect": {"speedup": 0.952}}))
    (tmp_path / "cool" / "ec2_three_way.json").write_text(json.dumps(
        {"chip_effect": {"speedup": 0.733}}))
    m = measurements_from_bench(tmp_path)
    assert m["sim2real_gap"]["value"] == 12.0 and "sim2real.json" in m["sim2real_gap"]["source"]
    assert m["cool_vs_stock"]["value"] == 0.952 and m["graviton_vs_x86"]["value"] == 0.733
    Report.model_validate(minimal(measurements=m))


def test_a_missing_benchmark_output_reads_as_not_measured(tmp_path):
    from blindspot.report import measurements_from_bench
    assert measurements_from_bench(tmp_path) == {
        "sim2real_gap": NOT_MEASURED, "cool_vs_stock": NOT_MEASURED, "graviton_vs_x86": NOT_MEASURED}
