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

Client: `AnthropicBedrockMantle` (the Messages-API endpoint on Bedrock), with
client-side refusal fallback enabled by default, since server-side fallbacks
are not available on Bedrock.
"""

from __future__ import annotations

import asyncio
import json

import anthropic

from .mcp_server import AgentSession, build_server

MODEL = "anthropic.claude-opus-5"
DOWNGRADE = {"anthropic.claude-opus-5": "anthropic.claude-sonnet-5",
             "anthropic.claude-sonnet-5": "anthropic.claude-haiku-4-5"}
REFUSAL_FALLBACK = "anthropic.claude-opus-4-8"
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
    return anthropic.AnthropicBedrockMantle(
        aws_region=region, aws_profile=profile, max_retries=0,
        middleware=[anthropic.BetaRefusalFallbackMiddleware([{"model": REFUSAL_FALLBACK}])],
    )


def _tools(server) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in asyncio.run(server.list_tools())]


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
    tools = _tools(server)
    messages = [{"role": "user", "content": (
        "Review the current envelope and the budget, then decide which axes to spend "
        "the remaining probes on and propose that. Explain the most important failure.")}]

    calls, overloads, status, summary = 0, 0, "completed", ""
    while True:
        if calls >= MAX_MODEL_CALLS:
            status = "call_cap_reached"
            break
        try:
            calls += 1
            response = client.beta.messages.create(
                model=model, max_tokens=16000, system=SYSTEM, tools=tools, messages=messages)
        except anthropic.OverloadedError:
            overloads += 1
            if overloads >= OVERLOAD_RETRIES:
                if model not in DOWNGRADE:
                    status = "overloaded"
                    break
                model, overloads = DOWNGRADE[model], 0
            continue

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
            "axis_order": session.order, "allocation": session.allocation,
            "proposals": session.proposals, "budget": session.contract.describe()}
