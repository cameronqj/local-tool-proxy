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
- Detects tool-using requests for configured compatibility models.
- Forces a reliable non-streaming upstream request for those tool turns.
- Rewrites common local-model tool-call formats into OpenAI-style `tool_calls`.
- Adds optional diagnostics for model tool-use collapse.
- Offers an experimental, opt-in stabilization mode that can retry one collapsed
  tool turn.

## What It Does Not Do

- It does not execute tools.
- It does not edit files.
- It does not replace an agent harness.
- It does not replace general routers such as LiteLLM.
- It does not guarantee that a weak or drifting model will finish a task.
- It does not log raw prompts by default.

## Quick Start

Install from a local checkout:

```bash
python3 -m pip install -e ".[dev]"
```

Start the proxy in front of Ollama:

```bash
local-tool-proxy \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b
```

If you only need local access, bind to loopback:

```bash
local-tool-proxy --host 127.0.0.1
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

The earlier development command name, `gptfixes`, is still installed as a
temporary compatibility alias. Public docs and examples use `local-tool-proxy`.

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

Run the mock rewrite test directly:

```bash
python3 -m proxy.test_rewrite_mock
```

That test starts a mock upstream and a local proxy, then verifies that an OpenAI
client receives proper `tool_calls`.

See [docs/testing.md](docs/testing.md) for live harness evaluation notes.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Use cases](docs/use-cases.md)
- [Experiment log](docs/experiment-log.md)
- [Testing](docs/testing.md)
- [Project constitution](docs/constitution.md)
- [ADRs](docs/adr/)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).
