# Use Cases

`local-tool-proxy` is useful when a model appears capable of tool use, but the
client cannot understand the exact wire format the model or upstream server
emits.

It is not a model-quality fix. If a model does not understand the task, ignores
tool instructions, or cannot sustain multi-step work, the proxy can make that
failure easier to observe but may not make the task succeed.

## OpenCode With Local Ollama Models

Some local models expose tool support through Ollama, but harnesses may fail to
consume the emitted tool calls. The proxy can sit between OpenCode and Ollama:

```text
OpenCode -> local-tool-proxy -> Ollama
```

Use this when you see symptoms such as:

- The model writes JSON tool calls in normal assistant content.
- The model emits XML-ish tool blocks.
- The client reports no tool use even though the raw model output contains tool intent.
- The client has trouble discovering local models.

Example proxy command:

```bash
local-tool-proxy \
  --host 127.0.0.1 \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx
```

Then configure the harness to use:

```text
http://localhost:9000/v1
```

## OpenAI SDK Clients

Custom Python or JavaScript clients often expect a strict OpenAI response shape.
The proxy can normalize common local-model variants into `tool_calls` before the
SDK sees the response.

This is useful for small experiments where you want to keep the client code
close to normal OpenAI SDK usage while swapping in a local upstream server.

## Aider, Continue, And Similar Harnesses

Any tool that can target an OpenAI-compatible base URL may be able to use the
proxy. Success depends on how strictly the harness parses responses and whether
it relies on streaming tool-call deltas.

For configured compatibility models, tool-using requests are sent upstream as
non-streaming requests so the proxy can inspect and rewrite the full response.

## Debugging Tool-Use Collapse

`observe` mode helps answer a simple question: did the model use a tool, describe
an intended action in prose, print literal commands, or produce a final answer?

This is useful when comparing prompts, models, or harnesses.

## Experimental Recovery

`stabilize` mode can perform one internal retry when the model appears to have
stopped using tools. This is intentionally conservative and opt-in.

Use it for experiments, not as proof that the proxy can make unreliable models
reliable.

## When Not To Use This

Use LiteLLM or another router when you need multi-provider routing, auth,
budgets, rate limits, or production-grade observability.

Use a model-specific harness integration when the client has native support for
the model's tool-call format.

Avoid this proxy for untrusted network deployment unless you have reviewed the
security posture and bound it appropriately.
