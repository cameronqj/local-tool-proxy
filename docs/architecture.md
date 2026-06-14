# Architecture

`local-tool-proxy` is a narrow OpenAI-compatible reverse proxy.

```text
client harness
  |
  | /v1/chat/completions
  v
local-tool-proxy
  |
  | /v1/chat/completions
  v
local model server
```

The main design goal is protocol repair, not orchestration.

## Request Flow

1. The client sends an OpenAI-compatible request to the proxy.
2. The proxy extracts the model name and checks whether tools are present.
3. If the model is not configured as a compatibility model, the request is passed through.
4. If the model is configured and tools are present, the proxy uses the compatibility path.
5. The compatibility path forces a non-streaming upstream request for reliability.
6. The proxy inspects the upstream response.
7. If assistant content contains tool intent, the proxy rewrites it into `tool_calls`.
8. The client receives an OpenAI-compatible response.

For ordinary requests without tools, or requests for models outside
`--compat-models`, the proxy should behave like a simple pass-through layer.

## Components

### `proxy/server.py`

FastAPI app, request routing, compatibility path, health endpoints, and CLI.

### `proxy/rewriters.py`

Pure parsing and synthesis logic. This is the core protocol-repair module.

Current strategies include:

- JSON in content.
- JSON-ish repair.
- `toolName{...}` fallback when known tool names are available.
- XML-ish tool blocks.
- Basic streaming accumulator scaffolding.

Parser behavior should fail closed: prose that merely resembles a tool call
should not become a tool call unless the strategy has enough evidence.

### `proxy/collapse.py`

Classifies assistant responses in tool-using contexts. Categories include
`tool_calls`, `tool_intent_prose`, `literal_commands`, `final_answer`, `chat`,
and `unclear`.

### `proxy/stabilizer.py`

Builds the optional retry payload for `--mode stabilize`. This is only active
when explicitly enabled.

### `proxy/planner.py`

Extracts lightweight milestones for recovery hints. It is not a task planner and
does not own execution.

## Public Endpoints

- `/health`: basic process health.
- `/v1/models`: proxied model listing.
- `/v1/chat/completions`: OpenAI-compatible chat completions path.

## Safety Boundary

The proxy rewrites model messages. It does not run commands, write files, inspect
the workspace, or decide task success.

## Current Tradeoff

For configured compatibility models, tool-using requests are forced to
non-streaming upstream calls. This improves reliability for small local models,
but means full streaming tool-call reconstruction remains future work.

Clients that depend on streaming tool-call deltas may need additional testing
before they are a good fit.
