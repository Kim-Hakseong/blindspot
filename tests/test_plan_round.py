"""One planning round of a cloud run (W4-4, W4-5).

`plan_round` is a pure function of the run definition and the probe ledger --
the ledger is the only state, so the planner can be killed and retried at any
point. Tests simulate whole runs against an oracle and compare with the local
search, then check the budget contract halts a run with its partial results.
"""

from __future__ import annotations

import pytest

from blindspot.boundary.locate import locate_on_axis
from blindspot.cloud.plan import plan_round
from blindspot.degrade import REGISTRY

BASELINE = 0.6
THR = {"motion_blur.exposure_ms": 13.1, "low_light.illuminance_lux": 19.0}


def run_def(budget=0.40, axes=("motion_blur.exposure_ms", "low_light.illuminance_lux")):
    out = []
    for axis_id in axes:
        deg, _, field = axis_id.partition(".")
        a = REGISTRY.get(deg).axis(field)
        out.append({"axis": axis_id, "unit": a.unit, "lo": a.lo, "hi": a.hi,
                    "severe_end": a.severe_end, "target_width": (a.hi - a.lo) / 32})
    return {"run_id": "r1", "axes": out, "verify_samples": 5, "budget_usd": budget,
            "cost_per_probe_usd": 0.01, "seed": 7}


def fake_probe(p):
    """Oracle: mAP drops below 60% of baseline past each axis's threshold."""
    if not p["set"]:
        return BASELINE
    (axis_id, value), = p["set"].items()
    deg, _, field = axis_id.partition(".")
    severe_hi = REGISTRY.get(deg).axis(field).severe_end == "hi"
    failed = value >= THR[axis_id] if severe_hi else value <= THR[axis_id]
    return 0.1 if failed else 0.55


def simulate(run):
    ledger, rounds = [], 0
    while True:
        step = plan_round(run, ledger)
        if step["action"] != "probe":
            return step, ledger, rounds
        rounds += 1
        for p in step["probes"]:
            ledger.append({"probe_id": p["probe_id"], "set": p["set"], "map50": fake_probe(p)})


def test_first_round_measures_the_baseline_alone():
    step = plan_round(run_def(), [])
    assert step["action"] == "probe"
    assert [p["probe_id"] for p in step["probes"]] == ["baseline"]


def test_a_full_run_finds_the_same_boundaries_as_the_local_search():
    step, ledger, _ = simulate(run_def())
    assert step["action"] == "done"
    for f in step["findings"]:
        deg, _, field = f["axis"].partition(".")
        axis = REGISTRY.get(deg).axis(field)
        failed = (lambda v, a=f["axis"]: fake_probe({"set": {a: v}}) < 0.6 * BASELINE)
        local = locate_on_axis(axis, failed, (axis.hi - axis.lo) / 32, verify_samples=5)
        assert f["status"] == local.status.value
        assert (f["lower"], f["upper"]) == pytest.approx((local.lower, local.upper))


def test_axes_are_probed_in_parallel_waves():
    _, _, rounds = simulate(run_def())
    # 1 baseline round + 1 scan wave + a few bisection rounds, not one per probe.
    assert rounds <= 8


def test_probe_ids_are_deterministic_and_unique():
    _, a, _ = simulate(run_def())
    _, b, _ = simulate(run_def())
    ids = [r["probe_id"] for r in a]
    assert ids == [r["probe_id"] for r in b]
    assert len(ids) == len(set(ids))


def test_budget_exhaustion_halts_with_partial_findings():
    step, ledger, _ = simulate(run_def(budget=0.08))
    assert step["action"] == "halted"
    assert step["state"] == "AWAITING_APPROVAL"
    assert len(ledger) <= 8, "spent past the contract"
    assert "budget" in step["reason"]
    assert step["spent_usd"] == pytest.approx(0.01 * len(ledger))


def test_a_partial_wave_is_cut_to_what_the_contract_affords():
    # 1 baseline + 5 + 5 scan points = 11 needed by round 2; afford only 8.
    step = None
    ledger = []
    run = run_def(budget=0.08)
    while True:
        step = plan_round(run, ledger)
        if step["action"] != "probe":
            break
        assert 0.01 * (len(ledger) + len(step["probes"])) <= 0.08 + 1e-9
        for p in step["probes"]:
            ledger.append({"probe_id": p["probe_id"], "set": p["set"], "map50": fake_probe(p)})


def test_findings_carry_reproduction_commands():
    step, _, _ = simulate(run_def())
    for f in step["findings"]:
        if f["status"] == "located":
            assert f["reproduce"].startswith("uv run blindspot probe --set ")
            assert "--seed 7" in f["reproduce"]


def test_an_axis_cap_stops_that_axis_as_capped_never_as_a_boundary():
    # An agent's accepted allocation caps probes per axis. A capped axis is
    # unresolved: it is reported, but with no bracket and no reproduce command.
    run = run_def() | {"probes_per_axis": {"motion_blur.exposure_ms": 3}}
    step, ledger, _ = simulate(run)
    blur = [r for r in ledger if list(r["set"]) == ["motion_blur.exposure_ms"]]
    assert len(blur) == 3
    f = next(x for x in step["findings"] if x["axis"] == "motion_blur.exposure_ms")
    assert f["status"] == "capped" and f["lower"] is None and f["upper"] is None
    assert f["reproduce"] is None and f["probes_used"] == 3
    other = next(x for x in step["findings"] if x["axis"] == "low_light.illuminance_lux")
    assert other["status"] == "located"


def test_prepaid_agent_cost_is_charged_before_any_probe():
    # 0.05 USD at 0.01/probe: 0.02 spent on the agent and 0.01 on the baseline leave 2.
    run = run_def(budget=0.05) | {"prepaid_usd": 0.02}
    step = plan_round(run, [{"probe_id": "baseline", "set": {}, "map50": BASELINE}])
    assert len(step["probes"]) == 2 and step["truncated"]
    assert step["spent_usd"] == pytest.approx(0.03)
