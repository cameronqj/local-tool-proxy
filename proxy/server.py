#!/usr/bin/env python3
"""
local-tool-proxy

Minimal in-line OpenAI-compatible proxy to improve tool-calling reliability
for small local models (Gemma 4 E4B/E2B via MLX, gpt-oss, etc.) when used
with normal harnesses such as OpenCode, Codex CLI, Aider, etc.

Current state: fully functional transparent reverse proxy.
Next: add model-specific rewriting for the common small-model failure modes
(JSON-in-content, imperfect streaming tool_calls deltas, etc.).

Usage:
    python3 -m proxy.server --port 9000 \
        --ollama-base http://localhost:11434/v1 \
        --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b

Then configure your harness to talk to http://localhost:9000/v1 instead of
the raw Ollama endpoint.
"""

import argparse
import json
import logging
import sys
import uuid
from typing import Set, Optional, Dict, Any

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask

# Import our small-model rewriters (the core fix logic)
try:
    from .rewriters import (
        parse_tool_call_from_content,
        looks_like_tool_call_json,
        synthesize_tool_call_response,
        rewrite_stream_chunk,
        is_tool_call_delta,
    )
except ImportError:
    # Allow running as python -m proxy.server from repo root
    from rewriters import (
        parse_tool_call_from_content,
        looks_like_tool_call_json,
        synthesize_tool_call_response,
        rewrite_stream_chunk,
        is_tool_call_delta,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local-tool-proxy")

# Global state - defined early so lifespan and other top-level code can reference them
OLLAMA_BASE: str = "http://localhost:11434/v1"
COMPAT_MODELS: Set[str] = set()
CLIENT = httpx.AsyncClient(timeout=600.0)  # long timeout for generation


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("local-tool-proxy starting")
    logger.info(f"  upstream: {OLLAMA_BASE}")
    logger.info(f"  compat models: {sorted(COMPAT_MODELS) or '(none)'}")
    logger.info("  Goal: Make stock OpenCode (and similar harnesses) work with small local models")
    logger.info("  via clean in-line translation of tool calling output.")
    yield
    # Shutdown
    await CLIENT.aclose()


app = FastAPI(title="local-tool-proxy", version="0.1.0-dev", lifespan=lifespan)


def is_compat_model(model: str) -> bool:
    if not model:
        return False
    model = model.lower()
    return any(m in model for m in COMPAT_MODELS)


async def _forward_request(request: Request, path: str):
    """Core transparent forwarder. Handles both streaming and non-streaming."""
    url = f"{OLLAMA_BASE.rstrip('/')}/{path.lstrip('/')}"

    # Read body once
    body = await request.body()

    # Try to extract model for logging / future compat decisions
    model = None
    if body:
        try:
            payload = json.loads(body)
            model = payload.get("model") or payload.get("model_name")
        except Exception:
            pass

    compat = is_compat_model(model) if model else False

    method = request.method
    headers = dict(request.headers)
    # Strip hop-by-hop headers that httpx will set itself
    for h in ("host", "connection", "keep-alive", "proxy-authenticate",
              "proxy-authorization", "te", "trailers", "transfer-encoding",
              "upgrade"):
        headers.pop(h, None)

    if compat:
        logger.info(f"COMPAT MODE  model={model}  method={method}  path=/{path}")

    # Forward the request
    try:
        upstream = await CLIENT.send(
            CLIENT.build_request(
                method=method,
                url=url,
                content=body,
                headers=headers,
            ),
            stream=True,   # always stream from upstream so we can decide later
        )
    except httpx.RequestError as e:
        logger.error(f"Upstream error to {url}: {e}")
        return Response(content=str(e), status_code=502)

    # Build response headers (important for SSE)
    resp_headers = dict(upstream.headers)
    # Remove hop-by-hop again
    for h in ("connection", "keep-alive", "transfer-encoding"):
        resp_headers.pop(h, None)

    if upstream.status_code >= 400:
        content = await upstream.aread()
        await upstream.aclose()
        logger.warning(f"Upstream {upstream.status_code} for model={model}")
        return Response(content=content, status_code=upstream.status_code, headers=resp_headers)

    # Streaming or regular?
    content_type = upstream.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type or "stream" in str(upstream.headers.get("transfer-encoding", "")).lower()

    if is_stream:
        async def stream_body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=content_type or "text/event-stream",
            background=BackgroundTask(upstream.aclose),
        )
    else:
        # Non-streaming — read everything then close
        content = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=resp_headers,
        )


# === Routes ===

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """
    Clean models endpoint. Helps stock OpenCode and other harnesses
    discover the models without falling back to 'ollama-cloud'.
    """
    models = []
    for m in sorted(COMPAT_MODELS):
        models.append({
            "id": m,
            "object": "model",
            "created": 1700000000,
            "owned_by": "local-tool-proxy",
        })
    return {"object": "list", "data": models}


@app.get("/health")
@app.get("/v1/health")
async def health():
    """Simple health check for scripts / monitoring before launching OpenCode."""
    return {
        "status": "ok",
        "upstream": OLLAMA_BASE,
        "compat_models": sorted(COMPAT_MODELS),
    }


def _has_tools(body: bytes) -> bool:
    """Quick check if this is a tool-using request (important for OpenCode)."""
    if not body:
        return False
    try:
        payload = json.loads(body)
        return bool(payload.get("tools"))
    except Exception:
        return False


def _looks_like_rigid_structured_prompt(messages: list) -> bool:
    """Heuristic to detect prompts that demand very strict output formats
    (numbered steps only, exact commit messages, 'show every command', etc.).
    This helps us log and potentially apply stronger formatting discipline.
    """
    if not messages:
        return False
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    rigid_signals = [
        "numbered", "exactly", "strict rules", "show every command",
        "no extra commentary", "only output", "in this exact order"
    ]
    text = last_user.lower()
    return any(sig in text for sig in rigid_signals)


# Explicit chat completions routes — MUST be defined BEFORE the catch-all
# so that /v1/chat/completions is handled by the special logic for stock OpenCode.
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    The critical path for stock OpenCode + small models.

    When we see a tool-using request for a compat model (Gemma 4 E4B etc.),
    we take the special _handle_tool_request_for_compat_model path.
    This is the main lever to make unmodified OpenCode work.
    """
    body = await request.body()

    model = None
    try:
        payload = json.loads(body)
        model = payload.get("model")
    except Exception:
        pass

    compat = is_compat_model(model) if model else False
    has_tools = _has_tools(body)

    if compat and has_tools:
        return await _handle_tool_request_for_compat_model(request, model, body)

    # Normal traffic (including non-tool requests for compat models)
    return await _forward_request(request, "chat/completions")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    """
    Transparent catch-all for everything else.
    """
    return await _forward_request(request, path)


async def _handle_tool_request_for_compat_model(
    request: Request,
    model: str,
    original_body: bytes,
) -> Response:
    """
    Special path for stock OpenCode + small models on tool-using turns.

    Strategy (pragmatic, high success rate on M4 Air):
    - Force non-streaming to the upstream (more reliable for Gemma 4 E* tool calling).
    - On the response, use rewriters.py to convert "JSON in content" into proper tool_calls.
    - Return a clean response that stock OpenCode's AI SDK can parse without issues.
    """
    # Detect rigid/structured prompts for better diagnostics and future format enforcement
    is_rigid = False
    try:
        payload_for_detection = json.loads(original_body)
        messages = payload_for_detection.get("messages", [])
        is_rigid = _looks_like_rigid_structured_prompt(messages)
        if is_rigid:
            logger.info(f"RIGID STRUCTURED PROMPT detected for {model} — extra format discipline may be needed")
    except Exception:
        pass

    trace_id = f"gptfixes-{uuid.uuid4().hex[:12]}"
    logger.info(f"[{trace_id}] COMPAT TOOL PATH for {model} — forcing non-stream + rewrite attempt (rigid={is_rigid})")

    # Phase 3 improvement: respect per-model preference for forcing non-stream on tool requests
    # (for now we key off the compat model list; later this can come from profiles)
    force_non_stream = model and any(m in model.lower() for m in COMPAT_MODELS)

    # Structured logging for the request (Phase 3)
    try:
        orig_payload = json.loads(original_body)
        tools_count = len(orig_payload.get("tools", [])) if orig_payload.get("tools") else 0
        stream_requested = orig_payload.get("stream", False)
    except Exception:
        tools_count = 0
        stream_requested = None

    logger.info(f"[{trace_id}] request: model={model} mode=compat stream_requested={stream_requested} tools_count={tools_count} rigid={is_rigid}")

    # Force non-streaming for reliability (Phase 3)
    try:
        payload = json.loads(original_body)
        payload["stream"] = False
        forced_body = json.dumps(payload).encode()
    except Exception:
        forced_body = original_body

    base = OLLAMA_BASE.rstrip("/")
    # Avoid double /v1 if the configured base already ends with it
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in
               ("host", "connection", "content-length")}

    try:
        upstream = await CLIENT.post(url, content=forced_body, headers=headers)
    except Exception as e:
        logger.error(f"Upstream error in compat tool path: {e}")
        if 'upstream' in locals():
            await upstream.aclose()
        return Response(content=str(e), status_code=502)

    if upstream.status_code >= 400:
        content = upstream.content
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code)

    # Now apply rewriting logic
    try:
        data = upstream.json()
    except Exception:
        content = upstream.content
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code)

    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {}) or {}
        content_str = msg.get("content") or ""
        finish = choices[0].get("finish_reason")

        if looks_like_tool_call_json(content_str):
            # Extract known tool names from the original request so the rewriter
            # can use ToolBridge-style name-guided parsing (big help for creative
            # "toolName{json}" or XML output from small models).
            tool_names: list[str] = []
            try:
                orig = json.loads(original_body)
                for t in (orig.get("tools") or []):
                    fn = t.get("function") or t
                    name = fn.get("name")
                    if name:
                        tool_names.append(name)
            except Exception:
                pass

            tool_calls = parse_tool_call_from_content(content_str, known_tool_names=tool_names or None)
            if tool_calls:
                logger.info(f"[{trace_id}] ✓ REWROTE tool call(s) for stock OpenCode: { [t['function']['name'] for t in tool_calls] }")
                rewritten = synthesize_tool_call_response(content_str, tool_calls, finish or "tool_calls")

                # Preserve important fields
                for key in ("id", "object", "created", "model", "usage", "system_fingerprint"):
                    if key in data:
                        rewritten[key] = data[key]
                rewritten["model"] = model

                await upstream.aclose()
                resp = JSONResponse(content=rewritten)
                resp.headers["x-gptfixes-trace-id"] = trace_id
                return resp

    # No rewrite happened — return original
    logger.info(f"[{trace_id}] No rewrite needed for {model} (rigid={is_rigid})")
    content = upstream.content
    await upstream.aclose()
    resp = Response(content=content, status_code=upstream.status_code)
    resp.headers["x-gptfixes-trace-id"] = trace_id
    return resp


# Note: The chat_completions route definition has been moved earlier in the file
# (before the catch-all) to ensure it takes precedence for /v1/chat/completions.


def main():
    parser = argparse.ArgumentParser(description="local-tool-proxy for small local models")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--ollama-base", default="http://localhost:11434/v1",
                        help="Base URL of the real Ollama OpenAI-compatible endpoint")
    parser.add_argument("--compat-models", default="gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b",
                        help="Comma-separated list of model substrings that get special small-model handling")
    args = parser.parse_args()

    global OLLAMA_BASE, COMPAT_MODELS
    OLLAMA_BASE = args.ollama_base
    COMPAT_MODELS = {m.strip().lower() for m in args.compat_models.split(",") if m.strip()}

    import uvicorn
    logger.info(f"Starting local-tool-proxy on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
