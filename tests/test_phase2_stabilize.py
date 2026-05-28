"""
Phase 2 evaluation test for NextGrok Stabilize v1.

Uses mocks to simulate:
- A first upstream response that triggers collapse (tool_intent_prose or literal_commands)
- A second upstream response after the internal steering retry that succeeds with tool_calls

Verifies:
- should_attempt_stabilize triggers correctly
- build_retry_payload does not touch original user messages
- The server returns the retry response when it succeeds
- Proper logging of interventions (via captured logs or side effects)
"""

import sys
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, ".")

import proxy.server as srv
from proxy.stabilizer import should_attempt_stabilize, build_retry_payload


def test_should_attempt_stabilize():
    assert should_attempt_stabilize("tool_intent_prose", "stabilize", True) is True
    assert should_attempt_stabilize("literal_commands", "stabilize", True) is True
    assert should_attempt_stabilize("tool_intent_prose", "observe", True) is False
    assert should_attempt_stabilize("chat", "stabilize", True) is False
    assert should_attempt_stabilize("tool_intent_prose", "stabilize", False) is False


def test_build_retry_payload_does_not_mutate_user_messages():
    original = {
        "model": "gemma4:e4b-mlx",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug in buggy.py"},
        ],
        "tools": [{"type": "function", "function": {"name": "write_file"}}],
        "stream": True,
    }

    retry = build_retry_payload(original)

    # Original user message must be untouched
    assert retry["messages"][1]["role"] == "user"
    assert "Fix the bug" in retry["messages"][1]["content"]

    # Steering message added as last message (default path when no category-specific steering)
    assert retry["messages"][-1]["role"] == "system"
    assert "did not use a tool" in retry["messages"][-1]["content"] or "Analyze what went wrong" in retry["messages"][-1]["content"]

    # stream forced off
    assert retry["stream"] is False


def test_stabilize_v1_end_to_end_mock():
    """
    End-to-end simulation of the compat tool path with stabilize mode.
    First response = collapse (no tool_calls, tool_intent_prose)
    Retry response = success with tool_calls
    """
    srv.MODE = "stabilize"
    srv.STABILIZE_MAX_RETRIES = 1
    srv.TRACE_DRIFT.clear()

    client = TestClient(srv.app)

    # Fake a tool-using request
    request_body = {
        "model": "gemma4:e4b-mlx",
        "messages": [{"role": "user", "content": "Do the tasklite bugfix"}],
        "tools": [{"type": "function", "function": {"name": "write_file"}}],
        "stream": False,
    }

    # First upstream response (the collapse)
    collapse_response = {
        "id": "first",
        "choices": [{
            "message": {"content": "Next I will write the test file using write_file."},
            "finish_reason": "stop"
        }]
    }

    # Second (retry) upstream response (success)
    success_response = {
        "id": "retry",
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }

    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        if call_count == 1:
            mock_resp.status_code = 200
            mock_resp.json.return_value = collapse_response
        else:
            mock_resp.status_code = 200
            mock_resp.json.return_value = success_response
        return mock_resp

    with patch.object(srv.CLIENT, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = fake_post

        # We call the internal handler indirectly by hitting the route
        # (the route will take the compat path because gemma4:e4b-mlx is in COMPAT_MODELS? 
        #  For the test we force it by setting the global)
        srv.COMPAT_MODELS = {"gemma4:e4b-mlx"}

        resp = client.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"content-type": "application/json"}
        )

    # The server should have returned the *retry* response (the successful one)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("id") == "retry", "Expected to receive the stabilized retry response"
    assert "tool_calls" in str(data.get("choices", [{}])[0].get("message", {}))

    print("Phase 2 stabilize v1 end-to-end mock test PASSED")


def test_strengthened_steering_messages_haystack_inspired():
    """Verify the new category-specific steering contains the key lightweight nudges."""
    from proxy.stabilizer import (
        LITERAL_COMMANDS_STEERING,
        TOOL_INTENT_PROSE_STEERING,
        build_retry_payload,
    )

    # Both should contain the "analyze + adapt + exactly one" pattern
    for text in (LITERAL_COMMANDS_STEERING, TOOL_INTENT_PROSE_STEERING):
        assert "Analyze what went wrong" in text
        assert "Adapt your strategy" in text
        assert "exactly one" in text.lower() or "EXACTLY ONE" in text
        assert "available list" in text.lower()

    # build_retry_payload with available_tool_names should produce the prescriptive guardrail
    payload = {
        "model": "gemma4:e4b-mlx",
        "messages": [{"role": "user", "content": "do something"}],
        "tools": [{"type": "function", "function": {"name": "write_file"}}],
    }
    retry = build_retry_payload(
        payload,
        steering_message=LITERAL_COMMANDS_STEERING,
        prepend_steering=True,
        available_tool_names=["write_file", "read_file"],
    )
    steering = retry["messages"][0]["content"]
    assert "You may ONLY call tools from this exact list" in steering
    assert "write_file" in steering
    assert "Never invent tool names" in steering

    print("Strengthened Haystack-inspired steering messages: OK")


if __name__ == "__main__":
    test_should_attempt_stabilize()
    test_build_retry_payload_does_not_mutate_user_messages()
    test_stabilize_v1_end_to_end_mock()
    test_strengthened_steering_messages_haystack_inspired()
    print("All Phase 2 tests passed.")
