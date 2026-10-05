"""The MCP server that exposes Blindspot to an agent (W6-1 .. W6-3).

Five tools. Two are read-only (the envelope, the facts behind a failure), one
runs a measurement under the budget contract, and two submit proposals that
the deterministic gate accepts or rejects. Every call lands in the decision
ledger, including the rejected ones.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

pytest.importorskip("mcp")

from blindspot.agent.mcp_server import AgentSession, build_server  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXES = ["motion_blur.exposure_ms", "low_light.illuminance_lux", "fog.beta_per_m", "jpeg.quality"]
needs_assets = pytest.mark.skipif(
    not (ROOT / "models" / "yolox_s.onnx").is_file()
    or not (ROOT / "val" / "road100" / "manifest.json").is_file(),
    reason="model or dataset not fetched")


def session(tmp_path, budget_probes=5):
    return AgentSession(
        dataset=ROOT / "val" / "road100", pipeline="yolox_s", frames=2, seed=7,
        axes=AXES, coverage={a: c for a, c in zip(AXES, [17.4, 88.4, 30.1, 23.9])},
        baseline_map50=0.61, threshold_map50=0.366,
        budget_usd=budget_probes * 0.01, cost_per_probe_usd=0.01,
        ledger_path=tmp_path / "decisions.jsonl", findings=[{"axis": AXES[0], "lower": 12.5, "upper": 13.75}],
    )


def call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    # MCPServer.call_tool returns content blocks and/or structured data.
    if isinstance(result, tuple):
        result = result[1] if result[1] is not None else result[0]
    if isinstance(result, dict):
        return result.get("result", result)
    text = result[0].text if isinstance(result, list) else result.content[0].text
    return json.loads(text)


def ledger(tmp_path):
    p = tmp_path / "decisions.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.is_file() else []


def test_exposes_exactly_the_five_tools(tmp_path):
    server = build_server(session(tmp_path))
    names = sorted(t.name for t in asyncio.run(server.list_tools()))
    assert names == sorted(["get_envelope", "probe_condition", "explain_failure",
                            "propose_axis_priority", "reallocate_budget"])


def test_get_envelope_reports_findings_budget_and_coverage(tmp_path):
    out = call(build_server(session(tmp_path)), "get_envelope", {})
    assert out["findings"][0]["axis"] == AXES[0]
    assert out["budget"]["probes_remaining"] == 5
    assert out["coverage_percent"][AXES[1]] == 88.4


def test_accepted_proposal_is_recorded(tmp_path):
    server = build_server(session(tmp_path))
    out = call(server, "propose_axis_priority",
               {"order": list(reversed(AXES)), "rationale": "jpeg is barely covered"})
    assert out["accepted_by_scheduler"] is True
    rec = ledger(tmp_path)[-1]
    assert rec["tool"] == "propose_axis_priority" and rec["accepted_by_scheduler"] is True


def test_rejected_proposal_is_recorded_too(tmp_path):
    server = build_server(session(tmp_path))
    out = call(server, "propose_axis_priority", {"order": AXES[:2], "rationale": "skip the rest"})
    assert out["accepted_by_scheduler"] is False
    assert ledger(tmp_path)[-1]["accepted_by_scheduler"] is False


def test_over_budget_reallocation_halts_for_approval(tmp_path):
    server = build_server(session(tmp_path, budget_probes=5))
    out = call(server, "reallocate_budget",
               {"probes_per_axis": {AXES[0]: 10}, "rationale": "need more resolution"})
    assert out["accepted_by_scheduler"] is False
    assert out["output"]["state"] == "AWAITING_APPROVAL"


@needs_assets
def test_probe_condition_measures_and_charges_the_contract(tmp_path):
    s = session(tmp_path, budget_probes=1)
    server = build_server(s)
    out = call(server, "probe_condition", {"axis": AXES[0], "value": 20.0})
    assert 0.0 <= out["map50"] <= 1.0
    assert out["failed"] == (out["map50"] < 0.366)
    assert out["judged_by"] == "boundary.criterion (deterministic)"
    refused = call(server, "probe_condition", {"axis": AXES[0], "value": 30.0})
    assert refused["state"] == "AWAITING_APPROVAL" and "map50" not in refused


def test_explain_failure_returns_facts_not_prose(tmp_path):
    """A1(c): the model writes the explanation; the tool supplies only facts."""
    out = call(build_server(session(tmp_path)), "explain_failure", {"axis": AXES[0]})
    assert out["axis"] == AXES[0] and out["boundary"] == {"lower": 12.5, "upper": 13.75}
    assert "explanation" not in out


def test_call_cap_switches_to_the_heuristic(tmp_path):
    server = build_server(session(tmp_path))
    for _ in range(10):
        call(server, "propose_axis_priority", {"order": AXES, "rationale": "r"})
    out = call(server, "propose_axis_priority", {"order": AXES, "rationale": "r"})
    assert out["accepted_by_scheduler"] is False and "heuristic" in out["output"]["reason"]


def test_a_real_stdio_client_can_list_and_call_tools(tmp_path):
    """W6-1 done condition: an MCP client talks to the server over stdio."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "blindspot.agent.mcp_server", "--ledger", str(tmp_path / "d.jsonl"),
              "--frames", "2"],
        cwd=str(ROOT),
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = await client.list_tools()
                result = await client.call_tool("get_envelope", {})
                return [t.name for t in tools.tools], result

    names, result = asyncio.run(run())
    assert "reallocate_budget" in names
    assert not getattr(result, "isError", False) and not getattr(result, "is_error", False)


def test_a_split_beyond_the_contract_is_kept_as_pending_not_applied(tmp_path):
    import asyncio
    from blindspot.agent.mcp_server import AgentSession, build_server
    axes = ["motion_blur.exposure_ms", "fog.beta_per_m"]
    s = AgentSession(dataset=tmp_path, pipeline="yolox_s", frames=1, seed=1, axes=axes, coverage={},
                     baseline_map50=0.6, threshold_map50=0.36, budget_usd=0.05,
                     cost_per_probe_usd=0.01, ledger_path=tmp_path / "d.jsonl")
    asyncio.run(build_server(s).call_tool(
        "reallocate_budget", {"probes_per_axis": {axes[0]: 9}, "rationale": "more blur"}))
    assert s.allocation is None and s.pending_allocation == {axes[0]: 9}
