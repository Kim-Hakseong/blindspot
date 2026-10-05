"""The agent loop: Claude on Amazon Bedrock driving Blindspot's MCP tools.

The tools the model sees are the MCP server's own, and every call is executed
by that server -- so the deterministic gate and the decision ledger see the
agent exactly as they see any other client. The loop adds the per-run rules
that belong to the model connection itself (rule A5):

- at most MAX_MODEL_CALLS model calls per run; after that the run continues
  on the fixed least-covered-first heuristic;
- an overload (HTTP 529) is retried, and after OVERLOAD_RETRIES failures on a
  model the loop steps down one model (DOWNGRADE);
- a refusal ends the agent's part of the run cleanly, on the heuristic.

Client: `AnthropicBedrock` (Bedrock's InvokeModel API) through the global
cross-region inference profiles. Model choice is what this account can call,
checked 2026-10-05: Haiku 4.5 answers; Sonnet 4.5 needs the account's one-time
Anthropic use-case form (a declaration only the account owner can make); the
Claude 5 family is "not available for this account". So the default is Haiku
4.5, the bottom of the chain, and an overload there ends the agent's part of
the run on the heuristic.

Token usage is priced (AWS Price List, Bedrock global on-demand, read
2026-10-05) so the run can charge it to its budget contract: run cost includes
the model calls (definition of run cost).
"""

from __future__ import annotations

import asyncio
import json

import anthropic

from ..cost.contract import BudgetExceeded
from .mcp_server import AgentSession, build_server

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
MODEL = HAIKU
DOWNGRADE = {SONNET: HAIKU}
#: USD per million (input, output) tokens.
PRICE_PER_MTOK = {HAIKU: (1.00, 5.00), SONNET: (3.00, 15.00)}
MAX_MODEL_CALLS = 10
OVERLOAD_RETRIES = 3

SYSTEM = (
    "You direct Blindspot, a tool that finds the capture conditions under which a "
    "vision pipeline starts failing. You can read the current envelope, measure "
    "single conditions under a fixed probe budget, propose the order in which axes "
    "are probed, and propose how the remaining probes are split across axes. "
    "Deterministic code accepts or rejects every proposal; you never decide pass or "
    "fail, where a boundary is, or what anything costs, and you cannot raise the "
    "budget. Prefer axes the validation set covers least. When you explain a "
    "failure, use only the facts explain_failure returns. Finish with a short "
    "summary of what you proposed and why."
)


def default_client(region: str = "us-east-1", profile: str = "blindspot"):  # pragma: no cover - AWS
    return anthropic.AnthropicBedrock(aws_region=region, aws_profile=profile, max_retries=0)


def _tools(server, measure_locally: bool = True) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in asyncio.run(server.list_tools())
            if measure_locally or t.name != "probe_condition"]


def _call(server, name: str, args: dict) -> tuple[str, bool]:
    try:
        result = asyncio.run(server.call_tool(name, args))
    except Exception as exc:  # unknown tool, bad arguments: tell the model
        return f"error: {exc}", True
    content = getattr(result, "content", None) or []
    text = "\n".join(getattr(c, "text", "") for c in content) or json.dumps(result, default=str)
    return text, bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def run_agent(session: AgentSession, client=None, model: str = MODEL) -> dict:
    client = client or default_client()
    server = build_server(session)
    tools = _tools(server, getattr(session, "measure_locally", True))
    messages = [{"role": "user", "content": (
        "Review the current envelope and the budget, then decide which axes to spend "
        "the remaining probes on and propose that. Explain the most important failure.")}]

    calls, overloads, status, summary = 0, 0, "completed", ""
    tokens_in = tokens_out = 0
    model_usd = 0.0
    while True:
        if calls >= MAX_MODEL_CALLS:
            status = "call_cap_reached"
            break
        try:
            calls += 1
            response = client.messages.create(
                model=model, max_tokens=4000, system=SYSTEM, tools=tools, messages=messages)
        except anthropic.OverloadedError:
            overloads += 1
            if overloads >= OVERLOAD_RETRIES:
                if model not in DOWNGRADE:
                    status = "overloaded"
                    break
                model, overloads = DOWNGRADE[model], 0
            continue

        usage = getattr(response, "usage", None)
        if usage is not None:
            pin, pout = PRICE_PER_MTOK[model]
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            call_usd = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
            model_usd += call_usd
            # Charged to the same contract the gate checks proposals against.
            try:
                session.contract.charge_usd(call_usd, "agent model call")
            except BudgetExceeded:
                status = "budget_exhausted"
                break

        if response.stop_reason == "refusal":
            status = "refused"
            break
        if response.stop_reason != "tool_use":
            summary = "".join(b.text for b in response.content if b.type == "text")
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            out, is_error = _call(server, block.name, dict(block.input))
            result = {"type": "tool_result", "tool_use_id": block.id, "content": out}
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})

    return {"status": status, "model": model, "model_calls": calls, "summary": summary,
            "input_tokens": tokens_in, "output_tokens": tokens_out, "model_usd": model_usd,
            "axis_order": session.order, "allocation": session.allocation,
            "pending_allocation": session.pending_allocation,
            "proposals": session.proposals, "budget": session.contract.describe()}
