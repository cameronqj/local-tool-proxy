"""End-to-end tests that pin the project's *public claims* about what the proxy
does and — just as importantly — does not do.

These run the real FastAPI app through `TestClient` with a mocked upstream, so
they exercise the actual `/v1/chat/completions` compat path rather than the
rewriter functions in isolation. No live model, no network, fully deterministic.

The claims under test (see README "Evidence level"):
  1. A malformed-but-recoverable tool call IS repaired into OpenAI `tool_calls`.
  2. Plain prose is NOT fabricated into a tool call (no intent is invented).
  3. Tool-ish but unparseable content is left untouched (no unsafe repair).
  4. Diagnostic traces record repair status/reason without leaking raw text.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import proxy.server as srv

COMPAT_MODEL = "gemma4:e4b-mlx"


def _upstream_response(content: str) -> httpx.Response:
    """Build an OpenAI-style upstream completion with everything stuffed in
    `content` and `tool_calls` left null — the small-model failure mode."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-mock",
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    )


def _post_tool_request(client: TestClient, tool_names):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": COMPAT_MODEL,
            "messages": [{"role": "user", "content": "Use a tool to do the task."}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": n, "description": n, "parameters": {}},
                }
                for n in tool_names
            ],
            "tool_choice": "auto",
        },
    )


@pytest.fixture
def compat_client():
    """A TestClient with the default (compat) configuration restored afterwards."""
    prev = (srv.COMPAT_MODELS, srv.MODE, srv.STABILIZE_MAX_RETRIES, srv.TRACE_FILE)
    srv.COMPAT_MODELS = {COMPAT_MODEL}
    srv.MODE = "compat"
    srv.STABILIZE_MAX_RETRIES = 0
    try:
        yield TestClient(srv.app)
    finally:
        (srv.COMPAT_MODELS, srv.MODE, srv.STABILIZE_MAX_RETRIES, srv.TRACE_FILE) = prev


def _tool_calls_of(resp) -> list:
    return resp.json()["choices"][0]["message"].get("tool_calls") or []


def test_malformed_but_recoverable_tool_call_is_repaired(compat_client):
    """Claim 1: JSON-in-content tool intent is repaired into real tool_calls."""
    upstream = _upstream_response(
        '{"name": "write_file", "arguments": {"path": "buggy.py", "content": "x"}}'
    )
    with patch.object(srv.CLIENT, "post", new_callable=AsyncMock, return_value=upstream):
        resp = _post_tool_request(compat_client, ["write_file"])

    assert resp.status_code == 200
    calls = _tool_calls_of(resp)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "buggy.py"


def test_plain_prose_is_not_fabricated_into_a_tool_call(compat_client):
    """Claim 2: when the model expresses no tool intent, none is invented.

    This is the core non-claim: the proxy repairs malformed intent, it does not
    *create* intent the model never expressed.
    """
    prose = (
        "I will write_file after I inspect the project. First I need to understand "
        "the existing tests, then I will decide what to change."
    )
    upstream = _upstream_response(prose)
    with patch.object(srv.CLIENT, "post", new_callable=AsyncMock, return_value=upstream):
        resp = _post_tool_request(compat_client, ["write_file"])

    assert resp.status_code == 200
    assert _tool_calls_of(resp) == []
    # Original prose is preserved unchanged for the harness to handle.
    assert resp.json()["choices"][0]["message"]["content"] == prose


def test_ambiguous_unparseable_content_is_not_unsafely_repaired(compat_client):
    """Claim 3: content that *looks* tool-ish but cannot be parsed is left as-is
    rather than guessed into a fabricated call."""
    ambiguous = "maybe {this is not valid json and matches no known tool"
    upstream = _upstream_response(ambiguous)
    with patch.object(srv.CLIENT, "post", new_callable=AsyncMock, return_value=upstream):
        resp = _post_tool_request(compat_client, ["write_file"])

    assert resp.status_code == 200
    assert _tool_calls_of(resp) == []
    assert resp.json()["choices"][0]["message"]["content"] == ambiguous


def test_trace_records_repair_status_and_reason(compat_client, tmp_path):
    """Claim 4: a repair emits a sanitized `rewrite` trace event recording the
    repair kind and recovered tool names, without leaking raw prompt/content."""
    trace_file = tmp_path / "trace.jsonl"
    srv.TRACE_FILE = trace_file

    upstream = _upstream_response('{"name": "write_file", "arguments": {"path": "a.py"}}')
    with patch.object(srv.CLIENT, "post", new_callable=AsyncMock, return_value=upstream):
        resp = _post_tool_request(compat_client, ["write_file"])

    assert resp.status_code == 200
    events = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    rewrite_events = [e for e in events if e["event"] == "rewrite"]
    assert len(rewrite_events) == 1
    rewrite = rewrite_events[0]
    assert rewrite["rewrite_kind"] == "content_to_tool_calls"
    assert rewrite["tool_names"] == ["write_file"]

    # Diagnostics must not carry raw prompt or model-output text.
    blob = trace_file.read_text()
    assert "buggy" not in blob
    assert "Use a tool to do the task" not in blob
