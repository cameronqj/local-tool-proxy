#!/usr/bin/env python3
"""
Simulated test for the proxy rewriter without needing real Ollama or Gemma.

This spins up:
- A tiny mock upstream (FastAPI) that returns "bad" responses
  (tool call as JSON inside the content field, exactly what Gemma 4 E4B often does).
- The real local-tool-proxy pointed at the mock.
- Then uses the official OpenAI client against the proxy and asserts
  that it received proper tool_calls in the response.

Run:
    python3 -m proxy.test_rewrite_mock

This validates the core rewrite logic that is intended to make stock OpenCode work.
"""

import asyncio
import json
import threading
import time
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
import httpx

# --- Mock upstream that always emits the "bad" pattern Gemma 4 often produces ---
mock_app = FastAPI()

@mock_app.post("/v1/chat/completions")
async def mock_chat():
    # Simulate what a small model frequently returns: tool call as plain JSON in content
    bad_response = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gemma4:e4b-mlx",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"name": "write_file", "arguments": {"path": "buggy.py", "content": "fixed code here"}}',
                "tool_calls": None
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    return JSONResponse(content=bad_response)


def run_mock_server(port: int):
    uvicorn.run(mock_app, host="127.0.0.1", port=port, log_level="warning")


@contextmanager
def mock_upstream(port: int = 11435):
    thread = threading.Thread(target=run_mock_server, args=(port,), daemon=True)
    thread.start()
    # Give it a moment to start
    time.sleep(1.2)
    try:
        # Return base without /v1 so the proxy's URL logic is exercised cleanly
        yield f"http://127.0.0.1:{port}"
    finally:
        # uvicorn thread will die with the process
        pass


def test_proxy_rewrites_json_in_content():
    """The actual test: proxy should turn the bad JSON-in-content into real tool_calls."""
    with mock_upstream(port=11436) as mock_base:
        # Import after potential previous runs
        import proxy.server as server_mod
        from proxy.server import app as proxy_app

        # Force the proxy globals BEFORE starting uvicorn
        server_mod.OLLAMA_BASE = mock_base
        server_mod.COMPAT_MODELS = {"gemma4:e4b-mlx"}

        proxy_port = 19002
        proxy_thread = threading.Thread(
            target=uvicorn.run,
            args=(proxy_app,),
            kwargs={"host": "127.0.0.1", "port": proxy_port, "log_level": "warning"},
            daemon=True
        )
        proxy_thread.start()
        time.sleep(1.8)  # give uvicorn time

        proxy_base = f"http://127.0.0.1:{proxy_port}/v1"

        client = OpenAI(base_url=proxy_base, api_key="dummy")

        # This request has "tools", so it should hit the special compat tool path
        resp = client.chat.completions.create(
            model="gemma4:e4b-mlx",
            messages=[{"role": "user", "content": "do something with tools"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                }
            }],
            tool_choice="auto",
        )

        msg = resp.choices[0].message
        print("finish_reason:", resp.choices[0].finish_reason)
        print("tool_calls present:", bool(msg.tool_calls))
        print("content:", repr(msg.content))

        assert msg.tool_calls is not None, "Proxy failed to rewrite JSON-in-content into tool_calls"
        assert len(msg.tool_calls) > 0
        assert msg.tool_calls[0].function.name == "write_file"

        print("\n✓ SUCCESS: Proxy correctly rewrote the small-model tool call output.")
        print("  This is the behavior we need for stock OpenCode + Gemma 4 E4B.")


if __name__ == "__main__":
    test_proxy_rewrites_json_in_content()