# Configuration

The main command is:

```bash
local-tool-proxy
```

## Common Options

```bash
local-tool-proxy \
  --host 127.0.0.1 \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b \
  --mode compat
```

Check the installed options at any time:

```bash
local-tool-proxy --help
```

## `--host`

Host to bind. The default is `127.0.0.1`, which keeps the proxy local to your
machine.

If another machine on a trusted network needs to reach the proxy, opt into a
broader bind address:

```bash
local-tool-proxy --host 0.0.0.0
```

## `--port`

Port exposed by the proxy. OpenAI-compatible clients should use:

```text
http://localhost:<port>/v1
```

## `--ollama-base`

Base URL for the upstream OpenAI-compatible local server. For Ollama this is
usually:

```text
http://localhost:11434/v1
```

## `--compat-models`

Comma-separated model substrings that should receive special tool-call handling.
Only tool-using requests for matching models enter the compatibility path.

Examples:

```bash
local-tool-proxy --compat-models gemma4:e4b-mlx
local-tool-proxy --compat-models gemma4:e4b-mlx,gpt-oss:20b
```

Requests for non-matching models are passed through without compatibility
rewriting.

## `--mode`

- `compat`: default transparent compatibility mode.
- `observe`: compatibility mode plus diagnostics.
- `stabilize`: compatibility mode plus one optional recovery retry.

Use `compat` first. Move to `observe` when you need to classify model behavior,
and to `stabilize` only when you are explicitly testing retry behavior.

## `--planner`

Only relevant with `--mode stabilize`.

- `disabled`: default.
- `observe`: reserved for planner diagnostics.
- `soft`: adds a short milestone hint to stabilization retry instructions.

The planner does not execute work or decide task success. It only affects retry
instructions in stabilization experiments.

## `--compat-streaming-rewrite`

Experimental and disabled by default.

When enabled, configured compatibility-model tool turns that request streaming
are sent upstream as streaming requests, buffered by the proxy, and rewritten
only if the completed stream contains parseable tool intent. If no confident
rewrite is possible, the original upstream stream is replayed.

Use this only when you are testing clients that require streaming. The normal
compatibility path still forces non-streaming tool turns because that is the
more reliable behavior for small local models.

## `--trace-file`

Writes sanitized JSONL trace events to the provided path:

```bash
local-tool-proxy --trace-file traces/local-tool-proxy.jsonl
```

Trace records include metadata such as trace id, model, mode, tool count,
rewrite type, collapse category, and stabilization attempt result. They do not
include raw prompts, tool schemas, or model output.

## `--debug-log-model-outputs`

Logs raw harness requests and model outputs. This is useful for experiments and
dangerous for privacy. Do not use with sensitive prompts or private source code.

Prefer leaving this disabled and promoting only sanitized findings into public
docs or issues.
