"""End-to-end smoke test against a fake upstream — no live model, no network.

This is the closest thing to "the real thing" without an LLM: a fake upstream
ASGI app stands in for the local model server, and the *real* proxy app sits in
front of it. Requests flow client -> proxy -> fake upstream -> proxy -> client
over an in-process ASGI transport (no sockets, no threads, no sleeps), so the
test is fully deterministic.

It exercises the work-plan smoke flow end to end:
  1. Fake upstream returns malformed-but-recoverable tool intent (JSON in
     content). The proxy repairs it into an OpenAI `tool_calls` response.
  2. Fake upstream returns plain prose. The proxy passes it through unchanged
     (no fabricated tool call).
  3. The sanitized trace records the repair and abstention reason codes.
"""

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import proxy.server as srv

COMPAT_MODEL = "fake-local"

# Scenario keyed by the user message content, so each case is explicit and the
# fake upstream stays a pure function of the request.
SCENARIOS = {
    "repair-json": '{"name": "write_file", "arguments": {"path": "out.py", "content": "x"}}',
    "prose": "I will look at the project first and then decide whether to write_file.",
}

fake_upstream = FastAPI()


@fake_upstream.get("/v1/models")
async def _models():
    return {"object": "list", "data": [{"id": COMPAT_MODEL, "object": "model"}]}


@fake_upstream.post("/v1/chat/completions")
async def _chat(request: Request):
    body = await request.json()
    key = body["messages"][-1]["content"]
    content = SCENARIOS[key]
    return JSONResponse(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": COMPAT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "write_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path"],
        },
    },
}


@pytest.fixture
def proxy_over_fake_upstream(tmp_path):
    """Wire the real proxy in front of the fake upstream over an ASGI transport."""
    prev = (
        srv.CLIENT,
        srv.OLLAMA_BASE,
        srv.COMPAT_MODELS,
        srv.MODE,
        srv.STABILIZE_MAX_RETRIES,
        srv.TRACE_FILE,
    )
    srv.CLIENT = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_upstream))
    srv.OLLAMA_BASE = "http://upstream/v1"
    srv.COMPAT_MODELS = {COMPAT_MODEL}
    srv.MODE = "compat"
    srv.STABILIZE_MAX_RETRIES = 0
    srv.TRACE_FILE = tmp_path / "trace.jsonl"
    try:
        yield TestClient(srv.app), srv.TRACE_FILE
    finally:
        (
            srv.CLIENT,
            srv.OLLAMA_BASE,
            srv.COMPAT_MODELS,
            srv.MODE,
            srv.STABILIZE_MAX_RETRIES,
            srv.TRACE_FILE,
        ) = prev


def _post(client, content_key):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": COMPAT_MODEL,
            "messages": [{"role": "user", "content": content_key}],
            "tools": [WRITE_FILE_TOOL],
            "tool_choice": "auto",
        },
    )


def test_end_to_end_repair_and_abstain_with_reason_codes(proxy_over_fake_upstream):
    client, trace_file = proxy_over_fake_upstream

    # 1. Malformed-but-recoverable tool intent -> repaired into tool_calls.
    repaired = _post(client, "repair-json")
    assert repaired.status_code == 200
    calls = repaired.json()["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "out.py"
    assert repaired.headers.get("x-local-tool-proxy-reason") == "repaired_json_content"

    # 2. Plain prose -> passed through unchanged, no fabricated call.
    prose = _post(client, "prose")
    assert prose.status_code == 200
    assert (prose.json()["choices"][0]["message"].get("tool_calls") or []) == []
    assert prose.json()["choices"][0]["message"]["content"] == SCENARIOS["prose"]
    assert prose.headers.get("x-local-tool-proxy-reason") == "abstain_no_tool_intent"

    # 3. Diagnostics carry both reason codes, without leaking raw text.
    events = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    reasons = {e.get("reason") for e in events if "reason" in e}
    assert "repaired_json_content" in reasons
    assert "abstain_no_tool_intent" in reasons
    blob = trace_file.read_text()
    assert "out.py" not in blob
    assert SCENARIOS["prose"] not in blob
