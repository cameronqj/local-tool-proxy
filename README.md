# local-tool-proxy

`local-tool-proxy` is an experimental OpenAI-compatible proxy for local model
tool-call compatibility.

It is built for a practical failure mode: a local model appears to be trying to
use tools, but emits the call in a shape the client does not understand. Examples
include JSON in assistant content, XML-ish tool blocks, `toolName{...}` snippets,
partial OpenAI-compatible fields, and other small-model output formats.

The proxy sits between a client such as OpenCode, Aider, Continue, or a custom
OpenAI SDK client and an upstream OpenAI-compatible local model server such as
Ollama.

## Status

This is a working prototype, not production infrastructure.

The core repair path is tested, including an integration-style mock upstream test
that turns Gemma-style JSON-in-content into proper OpenAI `tool_calls`. Real
harness testing has also shown the harder truth: a repair proxy can improve
protocol compatibility, but it does not make every model complete every agentic
task.

See [docs/experiment-log.md](docs/experiment-log.md) for the current evidence.

## What It Does

- Provides `/v1/chat/completions`, `/v1/models`, and health endpoints.
- Passes ordinary traffic through to the upstream model server.
- Proxies upstream model listing and marks configured compatibility models.
- Detects tool-using requests for configured compatibility models.
- Forces a reliable non-streaming upstream request for those tool turns.
- Rewrites common local-model tool-call formats into OpenAI-style `tool_calls`.
- Adds optional diagnostics for model tool-use collapse.
- Can write sanitized JSONL traces for request/rewrite/collapse events.
- Offers an experimental, opt-in stabilization mode that can retry one collapsed
  tool turn.

## What It Does Not Do

- It does not execute tools.
- It does not edit files.
- It does not replace an agent harness.
- It does not replace general routers such as LiteLLM.
- It does not guarantee that a weak or drifting model will finish a task.
- It does not log raw prompts by default.
- It does not create tool intent when the model did not express any. If the
  model returns plain prose, the proxy leaves it as prose.

## How It Differs From LiteLLM and General Routers

`local-tool-proxy` is a focused repair and diagnostics proxy, not a general LLM
gateway. General routers such as LiteLLM solve a broad problem — many providers,
auth, retries, load balancing, cost tracking, format translation across APIs.
They are excellent at that and this project does not try to replace them.

This proxy does one narrow thing instead: it sits in front of a single
OpenAI-compatible local server and repairs an important class of *malformed but
recoverable* tool-call output, while logging the tool-use failure modes it sees.

| | `local-tool-proxy` | General router (e.g. LiteLLM) |
| --- | --- | --- |
| Primary goal | Repair malformed local-model tool calls | Route/normalize across many providers |
| Scope | One upstream, OpenAI-compatible | Many providers and APIs |
| Tool-call repair | Core feature, tested | Not the focus |
| Failure-mode diagnostics | Core feature (collapse/drift traces) | Not the focus |
| Provider breadth, auth, billing | Out of scope | Core feature |

If you need broad routing, use a router. If a local model is *almost* emitting
tool calls and your harness just can't parse them, that gap is what this repairs.
The two can compose: point the router at this proxy, or this proxy at one model.

## Evidence Level

This project is deliberate about separating what is proven from what is not.

What is demonstrated by tests and reproducible runs:

- Malformed-but-recoverable tool intent (JSON-in-content, `toolName{...}`,
  XML-ish blocks) is repaired into valid OpenAI `tool_calls`.
- Plain prose and unparseable content are **not** turned into fabricated tool
  calls.
- Repairs and non-repairs are recorded as sanitized JSONL diagnostics that omit
  raw prompts and model output.

See [tests/test_claim_boundaries.py](tests/test_claim_boundaries.py) and the
[demo transcript](docs/demo.md), both of which run with no live model.

What is **not** claimed:

- This does not prove that a local model reliably completes real agentic tasks.
  End-to-end agent reliability must be measured separately — the captured
  OpenCode runs reached tool turns but did not pass artifact-level verification.
- The proxy improves protocol compatibility; it does not make a weak or drifting
  model finish a task.

The honest, full evidence trail — including the negative results — lives in
[docs/experiment-log.md](docs/experiment-log.md).

## Relationship to toolcall-repair-bench

This repo is the *proxy*. Measuring how well repair works across models and
malformed shapes is a separate concern, handled by its companion benchmark,
[`toolcall-repair-bench`](https://github.com/cameronqj/toolcall-repair-bench).
Keeping the benchmark separate is intentional: the proxy stays small and focused,
and the evidence it produces can be scrutinized on its own.

The intended pairing is to check both repos out under the same parent directory:

```text
parent/
  local-tool-proxy/        # this repo (the proxy under test)
  toolcall-repair-bench/   # the companion benchmark + leaderboard
```

```bash
# from the shared parent directory
git clone https://github.com/cameronqj/local-tool-proxy.git
git clone https://github.com/cameronqj/toolcall-repair-bench.git

# install this proxy editable so the benchmark can import/launch it
cd local-tool-proxy && python3 -m pip install -e ".[dev]" && cd ..

# run the benchmark against ../local-tool-proxy and generate the leaderboard
cd toolcall-repair-bench
# see that repo's README for exact commands; it expects the proxy at ../local-tool-proxy
```

> The benchmark commands above are illustrative. The authoritative, runnable
> steps live in the `toolcall-repair-bench` README; this repo does not vendor or
> execute the benchmark. The path it references is `../local-tool-proxy`.

## Quick Start

Install from a local checkout:

```bash
python3 -m pip install -e ".[dev]"
```

Start the proxy in front of Ollama:

```bash
local-tool-proxy \
  --host 127.0.0.1 \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b
```

The default bind host is `127.0.0.1`. If another machine on your trusted LAN
must reach the proxy, opt into a broader bind address:

```bash
local-tool-proxy --host 0.0.0.0
```

Point your OpenAI-compatible client at:

```text
http://localhost:9000/v1
```

Check that the proxy is up:

```bash
curl http://localhost:9000/health
curl http://localhost:9000/v1/models
```

## Demo (no Ollama required)

Want to see the repair happen without installing a model? Run the self-contained
demo. It starts a mock "local model" that emits the broken tool-call shapes small
models produce, puts the proxy in front of it, and shows the before/after for each:

```bash
make demo
# or: python3 demo.py
```

Everything runs in-process on localhost — no GPU, no model download. A captured
run with field-by-field explanation is in [docs/demo.md](docs/demo.md).

## Docker

Run the same no-Ollama demo in a container:

```bash
make docker-demo
```

Or serve the proxy from the container against a local Ollama:

```bash
make docker-run
# or, manually:
docker build -t local-tool-proxy .
docker run --rm -p 9000:9000 local-tool-proxy \
  --ollama-base http://host.docker.internal:11434/v1
```

Inside the container the proxy binds `0.0.0.0` so the published port is reachable
from the host. `host.docker.internal` resolves to the host's Ollama on Docker
Desktop (macOS/Windows); on Linux add `--add-host=host.docker.internal:host-gateway`.
Review [SECURITY.md](SECURITY.md) before exposing the proxy beyond your machine.

## Minimal SDK Example

Any client that accepts an OpenAI-compatible base URL can point at the proxy. For
example, with the Python OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="local")

response = client.chat.completions.create(
    model="gemma4:e4b-mlx",
    messages=[{"role": "user", "content": "List the files in the current project."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ],
)

print(response.choices[0].message.tool_calls)
```

The proxy only rewrites the model response. Your client or harness still owns
tool execution.

## Modes

`compat` is the default mode:

```bash
local-tool-proxy --mode compat
```

`observe` keeps compatibility behavior and adds collapse diagnostics:

```bash
local-tool-proxy --mode observe
```

`stabilize` is experimental. It can perform one internal upstream retry when the
model appears to have stopped using tools and started describing actions in
prose:

```bash
local-tool-proxy --mode stabilize --stabilize-max-retries 1
```

The optional soft planner only affects stabilization retry hints:

```bash
local-tool-proxy --mode stabilize --planner soft
```

For sanitized, machine-readable diagnostics, add a trace file:

```bash
local-tool-proxy --trace-file traces/local-tool-proxy.jsonl
```

Trace events include request metadata, rewrite decisions, collapse categories,
and stabilization attempts. They intentionally do not include raw prompts or
model output. Use `--debug-log-model-outputs` only for local debugging with
non-sensitive data.

Streaming tool-call rewriting is available as a separate experimental flag:

```bash
local-tool-proxy --compat-streaming-rewrite
```

When enabled, streaming compat tool turns are buffered. If the completed stream
contains parseable tool intent, the proxy emits a tool-call stream; otherwise it
replays the upstream stream unchanged. The default compatibility path remains
non-streaming for reliability.

Experimental modes are opt-in by design. See
[docs/constitution.md](docs/constitution.md) and
[docs/adr/0003-stabilize-mode-is-opt-in.md](docs/adr/0003-stabilize-mode-is-opt-in.md).

## Use Cases

- Running stock OpenCode against local Ollama models that emit incompatible tool
  calls.
- Debugging whether a model is producing tool intent, malformed tool calls, or
  plain prose.
- Normalizing JSON/XML-ish tool-call output for OpenAI SDK clients.
- Comparing small local models on real agentic tasks.
- Building a narrow compatibility layer before deciding whether a heavier router
  is needed.

See [docs/use-cases.md](docs/use-cases.md) for examples and limitations.

## Development

Run the test suite:

```bash
python3 -m pytest
```

Run linting:

```bash
python3 -m ruff check .
```

Run the mock rewrite integration test directly:

```bash
python3 -m pytest tests/test_integration_mock.py
```

That test starts a mock upstream and a local proxy, then verifies that an OpenAI
client receives proper `tool_calls`.

See [docs/testing.md](docs/testing.md) for live harness evaluation notes.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Demo transcript](docs/demo.md)
- [Use cases](docs/use-cases.md)
- [Experiment log](docs/experiment-log.md)
- [Testing](docs/testing.md)
- [Project constitution](docs/constitution.md)
- [ADRs](docs/adr/)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).
