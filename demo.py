#!/usr/bin/env python3
"""
Self-contained demo for local-tool-proxy — no Ollama, no GPU, no model download.

What it does:

  1. Starts a mock "local model" upstream that emits the exact broken tool-call
     shapes small local models produce (JSON-in-content, toolName{...}, XML tool
     blocks) — plus a plain-prose case that must NOT be treated as a tool call.
  2. Starts the real proxy in front of it.
  3. For each case, shows what the raw model emitted vs. what your harness
     receives after the proxy repairs it.

Everything runs in-process on localhost.

Run:
    make demo
    # or
    python3 demo.py
"""

import json
import socket
import sys
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

MODEL = "demo-local"
# Filled in at runtime with free ports so the demo never collides with a
# real proxy or anything else already listening.
PORTS = {"mock": 0, "proxy": 0}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# --- ANSI helpers (no-op when not a TTY) -----------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def bold(t):
    return _c("1", t)


def green(t):
    return _c("32", t)


def red(t):
    return _c("31", t)


def yellow(t):
    return _c("33", t)


def dim(t):
    return _c("2", t)


# --- The cases: real broken shapes a small model emits ----------------------
# `expect`: the tool name the proxy should recover, or None when the output is
# genuine prose that must be left alone (the precision case).
CASES = [
    {
        "label": "JSON object in content",
        "raw": '{"name": "write_file", "arguments": '
               '{"path": "buggy.py", "content": "fixed code here"}}',
        "tools": ["write_file"],
        "expect": "write_file",
    },
    {
        "label": "toolName{...} snippet",
        "raw": "run_terminal_cmd{'command': 'pytest -q'}",
        "tools": ["run_terminal_cmd"],
        "expect": "run_terminal_cmd",
    },
    {
        "label": "XML <tool_code> block",
        "raw": '<tool_code>run_command("mkdir multi_tool_test")</tool_code>',
        "tools": ["run_command"],
        "expect": "run_command",
    },
    {
        "label": "Plain prose (must NOT be turned into a tool call)",
        "raw": "I will write_file after I inspect the project. First I need to "
               "understand the existing tests, then I will decide what to change.",
        "tools": ["write_file"],
        "expect": None,
    },
]

# Shared between the main thread and the mock server thread (same process).
CURRENT = {"i": 0}

mock_app = FastAPI()


@mock_app.get("/__ping")
async def _ping():
    return {"ok": True}


@mock_app.post("/v1/chat/completions")
async def _mock_chat():
    """Return the current case's raw output the way a small model really does:
    everything stuffed in `content`, with `tool_calls` left null."""
    raw = CASES[CURRENT["i"]]["raw"]
    return JSONResponse(
        {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": raw, "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )


def _serve(app, port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait_for(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except Exception as e:  # not up yet
            last = e
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last}")


def _start_servers() -> None:
    import logging

    # Keep the demo output clean — the proxy's own INFO logs and httpx request
    # logs are not the point here.
    for name in ("local-tool-proxy", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    PORTS["mock"] = _free_port()
    PORTS["proxy"] = _free_port()

    # Mock upstream
    threading.Thread(target=_serve, args=(mock_app, PORTS["mock"]), daemon=True).start()

    # Real proxy, pointed at the mock, with our demo model marked as compat.
    import proxy.server as server

    server.OLLAMA_BASE = f"http://127.0.0.1:{PORTS['mock']}"
    server.COMPAT_MODELS = {MODEL}
    threading.Thread(target=_serve, args=(server.app, PORTS["proxy"]), daemon=True).start()

    _wait_for(f"http://127.0.0.1:{PORTS['mock']}/__ping")
    _wait_for(f"http://127.0.0.1:{PORTS['proxy']}/health")


def _request(base: str, case: dict) -> dict:
    resp = httpx.post(
        f"{base}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Use a tool to do the task."}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": name, "description": name, "parameters": {}},
                }
                for name in case["tools"]
            ],
            "tool_choice": "auto",
        },
        timeout=30.0,
    )
    return resp.json()


def _summarize_tool_calls(message: dict) -> str:
    calls = message.get("tool_calls")
    if not calls:
        return "(no tool_calls — left as assistant content)"
    out = []
    for c in calls:
        fn = c.get("function", {})
        args = fn.get("arguments", "")
        try:
            args = json.dumps(json.loads(args), separators=(", ", ": "))
        except Exception:
            pass
        out.append(f"{fn.get('name')}({args})")
    return ", ".join(out)


def main() -> int:
    print(bold("\nlocal-tool-proxy demo") + dim("  (mock upstream — no Ollama)\n"))
    _start_servers()

    mock_base = f"http://127.0.0.1:{PORTS['mock']}"
    proxy_base = f"http://127.0.0.1:{PORTS['proxy']}"

    passed = 0
    for i, case in enumerate(CASES):
        CURRENT["i"] = i

        raw = _request(mock_base, case)["choices"][0]["message"].get("content")
        proxied_msg = _request(proxy_base, case)["choices"][0]["message"]
        recovered = proxied_msg.get("tool_calls")
        recovered_name = recovered[0]["function"]["name"] if recovered else None
        ok = recovered_name == case["expect"]
        passed += ok

        print(bold(f"[{i + 1}] {case['label']}"))
        print(f"    {dim('model emitted   :')} {yellow(raw)}")
        print(f"    {dim('harness receives:')} {green(_summarize_tool_calls(proxied_msg))}")
        if case["expect"] is None:
            verdict = "left as prose (precision preserved)" if ok else "WRONGLY fabricated a tool call"
        else:
            verdict = f"repaired into {case['expect']}" if ok else f"expected {case['expect']}"
        print(f"    {(green('✓ ') if ok else red('✗ ')) + verdict}\n")

    line = f"{passed}/{len(CASES)} cases behaved as expected"
    print(bold(green(line)) if passed == len(CASES) else bold(red(line)))
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
