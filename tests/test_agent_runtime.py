"""The agent loop: Claude drives the MCP tools, under rules A1-A5.

Tested with a scripted fake client, so no model is called. The properties
pinned are the ones the rules require: at most ten model calls per run, a
one-step model downgrade after repeated overloads, a refusal ends the loop
cleanly, and every tool call still goes through the same server -- so the
gate and the decision ledger see it.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("mcp")

import anthropic  # noqa: E402

from blindspot.agent.mcp_server import AgentSession  # noqa: E402
from blindspot.agent.runtime import DOWNGRADE, MODEL, PRICE_PER_MTOK, run_agent  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXES = ["motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m", "jpeg.quality"]


def session(tmp_path):
    return AgentSession(
        dataset=ROOT / "val" / "road100", pipeline="yolox_s", frames=2, seed=7, axes=AXES,
        coverage={a: c for a, c in zip(AXES, [17.4, 88.4, 30.1, 23.9])},
        baseline_map50=0.61, threshold_map50=0.366, budget_usd=0.05, cost_per_probe_usd=0.01,
        ledger_path=tmp_path / "d.jsonl", findings=[])


def tool_use(name, args, id_="t1"):
    return NS(type="tool_use", name=name, input=args, id=id_)


def text(s):
    return NS(type="text", text=s)


class FakeClient:
    """Returns scripted responses; records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self
        self.beta = NS(messages=self)

    def create(self, **kw):
        self.requests.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def resp(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), stop_details=None)


def overloaded():
    return anthropic.OverloadedError("overloaded", response=NS(status_code=529, headers={},
                                                                request=None), body=None)


def ledger(tmp_path):
    return [json.loads(l) for l in (tmp_path / "d.jsonl").read_text().splitlines()]


def test_tool_calls_run_through_the_server_and_reach_the_ledger(tmp_path):
    client = FakeClient([
        resp("tool_use", tool_use("propose_axis_priority",
                                  {"order": list(reversed(AXES)), "rationale": "jpeg least covered"})),
        resp("end_turn", text("done")),
    ])
    out = run_agent(session(tmp_path), client=client)
    assert out["status"] == "completed" and out["model_calls"] == 2
    assert ledger(tmp_path)[0]["tool"] == "propose_axis_priority"
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"


def test_the_model_sees_exactly_the_mcp_tools(tmp_path):
    client = FakeClient([resp("end_turn", text("nothing to do"))])
    run_agent(session(tmp_path), client=client)
    names = sorted(t["name"] for t in client.requests[0]["tools"])
    assert names == sorted(["get_envelope", "probe_condition", "explain_failure",
                            "propose_axis_priority", "reallocate_budget"])
    assert client.requests[0]["model"] == MODEL


def test_call_cap_stops_the_loop_and_falls_back(tmp_path):
    """A5: at most ten model calls per run, then the fixed heuristic."""
    loop = [resp("tool_use", tool_use("get_envelope", {}, f"t{i}")) for i in range(20)]
    client = FakeClient(loop)
    out = run_agent(session(tmp_path), client=client)
    assert out["model_calls"] == 10 and len(client.requests) == 10
    assert out["status"] == "call_cap_reached"
    assert out["axis_order"] == ["motion_blur.exposure_ms", "jpeg.quality",
                                 "fog.beta_per_m", "low_light.illuminance_lux"]


def test_repeated_overload_downgrades_the_model_one_step(tmp_path):
    """A5: three overloads on a model, then the next model down."""
    upper = next(iter(DOWNGRADE))
    client = FakeClient([overloaded(), overloaded(), overloaded(), resp("end_turn", text("ok"))])
    out = run_agent(session(tmp_path), client=client, model=upper)
    assert out["status"] == "completed"
    assert [r["model"] for r in client.requests] == [upper] * 3 + [DOWNGRADE[upper]]
    assert out["model"] == DOWNGRADE[upper]


def test_overload_on_the_last_model_ends_on_the_heuristic(tmp_path):
    assert MODEL not in DOWNGRADE  # the default is the bottom of the chain
    out = run_agent(session(tmp_path), client=FakeClient([overloaded()] * 3))
    assert out["status"] == "overloaded" and out["axis_order"]


def test_a_refusal_ends_the_run_on_the_heuristic(tmp_path):
    r = resp("refusal")
    r.stop_details = NS(category=None, explanation="declined")
    out = run_agent(session(tmp_path), client=FakeClient([r]))
    assert out["status"] == "refused" and out["axis_order"]


def test_a_tool_error_is_returned_to_the_model_not_raised(tmp_path):
    client = FakeClient([
        resp("tool_use", tool_use("no_such_tool", {})),
        resp("end_turn", text("ok")),
    ])
    run_agent(session(tmp_path), client=client)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_token_usage_is_priced_so_the_run_contract_can_be_charged(tmp_path):
    a = resp("tool_use", tool_use("get_envelope", {}))
    a.usage = NS(input_tokens=1000, output_tokens=200)
    b = resp("end_turn", text("done"))
    b.usage = NS(input_tokens=3000, output_tokens=100)
    out = run_agent(session(tmp_path), client=FakeClient([a, b]))
    pin, pout = PRICE_PER_MTOK[MODEL]
    assert (out["input_tokens"], out["output_tokens"]) == (4000, 300)
    assert out["model_usd"] == pytest.approx((4000 * pin + 300 * pout) / 1e6)


def test_in_cloud_mode_the_agent_cannot_measure_on_this_machine(tmp_path):
    # Measurements of a cloud run happen on AWS Batch; the agent only proposes.
    s = session(tmp_path)
    s.measure_locally = False
    client = FakeClient([resp("end_turn", text("ok"))])
    run_agent(s, client=client)
    names = {t["name"] for t in client.requests[0]["tools"]}
    assert "probe_condition" not in names and "reallocate_budget" in names


def test_each_model_call_is_charged_as_it_happens_so_the_gate_sees_it(tmp_path):
    # 0.05 USD at 0.01/probe. The first call costs 0.02 (20k input tokens on
    # Haiku), so a 5-probe split no longer fits and the gate refuses it.
    s = session(tmp_path)
    first = resp("tool_use", tool_use("reallocate_budget",
                                      {"probes_per_axis": {AXES[0]: 5}, "rationale": "r"}))
    first.usage = NS(input_tokens=20000, output_tokens=0)
    out = run_agent(s, client=FakeClient([first, resp("end_turn", text("ok"))]))
    assert s.contract.spent_usd == pytest.approx(0.02)
    assert out["allocation"] is None and out["pending_allocation"] == {AXES[0]: 5}


def test_model_spend_beyond_the_contract_stops_the_agent(tmp_path):
    s = session(tmp_path)  # 0.05 USD
    big = resp("tool_use", tool_use("get_envelope", {}))
    big.usage = NS(input_tokens=60000, output_tokens=0)  # 0.06 USD
    out = run_agent(s, client=FakeClient([big, resp("end_turn", text("never reached"))]))
    assert out["status"] == "budget_exhausted" and out["model_calls"] == 1


def test_every_model_call_leaves_a_ledger_record_with_its_charge(tmp_path):
    a = resp("tool_use", tool_use("get_envelope", {}))
    a.usage = NS(input_tokens=1000, output_tokens=200)
    b = resp("end_turn", text("done"))
    b.usage = NS(input_tokens=3000, output_tokens=100)
    run_agent(session(tmp_path), client=FakeClient([a, b]))
    calls = [r for r in ledger(tmp_path) if r["tool"] == "agent.model_call"]
    pin, pout = PRICE_PER_MTOK[MODEL]
    assert [c["input"]["call"] for c in calls] == [1, 2]
    assert calls[0]["output"]["usd"] == pytest.approx((1000 * pin + 200 * pout) / 1e6)
    assert calls[1]["output"]["contract_spent_after_usd"] == pytest.approx(
        (4000 * pin + 300 * pout) / 1e6)
    assert all(c["accepted_by_scheduler"] for c in calls)
