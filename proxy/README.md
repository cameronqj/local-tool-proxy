# local-tool-proxy

Minimal in-line OpenAI-compatible proxy to make small local models (Gemma 4 E4B/E2B, gpt-oss, etc.) work reliably with normal "opencode-like" agent harnesses (OpenCode, Codex CLI, Aider, Continue, custom OpenAI-client agents, etc.).

## The Problem It Solves

Small efficient models on consumer Apple Silicon (especially via Ollama MLX) are often perfectly capable of tool use, but they (or the Ollama OpenAI-compat layer) emit output that popular harnesses don't recognize:

- Structured JSON or XML-ish tool calls appear in `content` instead of proper `tool_calls` objects.
- Streaming deltas are close but not perfect (extra fields like `reasoning_content`, split arguments, legacy `function_call`, etc.).
- Tool name / ontology mismatches (e.g. gpt-oss container.* world vs. what the harness was prompted for).

Result: harnesses fall back to "I can't use tools" or loop / reconnect even when the raw model is doing the right thing.

This proxy sits transparently between the harness and Ollama. Unmodified harnesses point at the proxy's `/v1` endpoint exactly like they would a normal OpenAI-compatible provider.

## Current Status (Ready for real hardware test)

- [x] Transparent reverse proxy (forwards everything, including streaming SSE).
- [x] Special non-streaming + rewriting path for tool-using turns on compat models (the main lever for stock OpenCode).
- [x] `/v1/models` + `/health` endpoints (easy discovery and readiness checks).
- [x] Core JSON-in-content → proper `tool_calls` rewrite working (proven in simulation against real Gemma-style bad output).
- [x] Clean startup (no deprecation warnings after lifespan migration).
- [x] Convenient launcher: `./proxy/start.sh`
- [ ] Full streaming delta reconstruction (basic accumulator exists; non-stream path is the recommended reliable mode for small models today).
- [ ] Additional parser types (XML, legacy function_call, etc.).

**Simulation test (2026-05-26)**: Full OpenCode-like tool request loop against a mock "Gemma 4" that emits JSON in content → proxy correctly rewrote it to proper `tool_calls`. Test passed cleanly.

The explicit goal is: **stock, unmodified OpenCode** can complete real coding tasks (our clean Task 01/02) using `gemma4:e4b-mlx` when pointed at this proxy on your 24 GB fanless M4 Air.

See `TESTING.md` for the exact commands and prompt to use on your real machine.

## Quick Start (once running)

```bash
# 1. Install deps (one time)
python3 -m pip install fastapi uvicorn httpx

# 2. Run the proxy (pointing at your real Ollama)
python3 -m proxy.server --port 9000 --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b

# 3. Configure your harness to use the proxy instead of raw Ollama
# Example for opencode.json (add a new provider):
```

```json
{
  "provider": {
    "local-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Proxy (small models fixed)",
      "options": {
        "baseURL": "http://localhost:9000/v1"
      },
      "models": {
        "gemma4:e4b-mlx": { "name": "Gemma 4 E4B (via proxy)", "tools": true }
      }
    }
  }
}
```

Then run OpenCode with `-m local-proxy/gemma4:e4b-mlx` (or equivalent) and give it a real coding task.

## Design Goals

- Extremely lightweight on 24 GB fanless M4 Air (the proxy itself should be invisible compared to model inference).
- Zero changes required to the harness (true "in-line" fix).
- Complements (does not replace) LiteLLM. Use LiteLLM when you want a big universal router; use this when you want targeted, observable fixes for the small-model-on-Apple-Silicon case + excellent debug logs.
- Easy to extend with new per-model adapters.
- Success metric: one of the clean minimal tasks in `../tasks/` (scaffold CLI or fix buggy longest-word) completes end-to-end via unmodified harness + this proxy + Gemma 4 E4B.

## Relationship to Other Work

- OpenCode PR #16531 (unmerged as of May 2026): The `toolParser` compat layer inside OpenCode. This proxy can be seen as an external version of the same idea that works for *any* harness.
- SmallHarness: Excellent client-side JSON/XML fallback parsers. We are stealing the best detection heuristics and moving them server-side into the proxy.

## Next Steps (for this repo)

See the todo list in the parent conversation and the code comments in `server.py`.

Contributions / experiments welcome — the whole point is to make small, fast local models practically usable for agentic coding on everyday hardware.
