# Experiment Log

This is the public experiment trail behind `local-tool-proxy`. It intentionally
keeps the claim narrow: protocol repair is useful and testable, while real
local-agent task completion still needs stronger evidence.

## Setup

- Hardware: 24 GB fanless Apple Silicon MacBook Air.
- Local model server: Ollama.
- Main model family under test: Gemma 4 E-series MLX variants, with some gpt-oss
  and QAT/MTP follow-up work.
- Harnesses explored: OpenCode, Codex CLI, raw OpenAI-compatible clients, and
  small mock harnesses.

## Baseline Finding

Small local models were fast enough to be interesting, but agentic coding was
fragile. Failures clustered around the protocol boundary:

- Model discovery issues in local harness configuration.
- Tool calls emitted in a shape the client did not consume.
- Tool names that matched a different harness ontology.
- Models drifting into prose or literal command text instead of calling tools.

The original runs also showed a measurement trap: a harness can exit cleanly
while the requested files, tests, or git history are missing from the isolated
verification workspace. Public results should therefore separate "the agent
loop returned" from "the task artifact was actually produced."

## Key Hypothesis

The model is often not completely incapable of tool use. In many cases, it emits
usable tool intent in the wrong representation.

That suggests a narrow repair proxy can help:

```text
malformed local-model tool intent -> local-tool-proxy -> OpenAI-compatible tool_calls
```

## Prototype Result

The mock rewrite test validates the central mechanism in a controlled setting:

- A mock upstream returns a Gemma-style tool call as JSON inside assistant content.
- The proxy detects the content.
- The proxy synthesizes a proper OpenAI `tool_calls` response.
- An OpenAI SDK client receives a normal tool call.

Current local verification:

```text
20 passed
```

That is the strongest positive result so far. It demonstrates that the proxy can
repair a specific, common wire-format failure without requiring a patched client.

## Live Ollama Smoke Test

On 2026-06-13, the current proxy branch was smoke-tested against a local Ollama
server with `gemma4:e4b-mlx`.

What passed:

- `/v1/models` proxied the live Ollama model list and marked configured
  compatibility models.
- A non-streaming tool request through the proxy returned native OpenAI-style
  `tool_calls`.
- `--trace-file` wrote sanitized JSONL request/collapse metadata without raw
  prompt or model-output text.
- With `--compat-streaming-rewrite` enabled, a streamed tool request completed
  with streamed `tool_calls`. Because Ollama already emitted native tool-call
  deltas, the proxy replayed the upstream stream rather than rewriting it.

This is a live integration smoke test, not a benchmark and not an artifact-level
agent success result. It confirms that the updated proxy surface still works
against an actual local model server.

## Real Harness Result

Real OpenCode evaluations with `gemma4:e4b-mlx` showed partial progress, not a
finished solution:

- The proxy compatibility path was entered.
- Tool turns were observed.
- Drift/collapse diagnostics were recorded.
- Some runs exited cleanly.
- The stabilize mode recorded at least one internal retry on a hard rigid prompt
  after detecting literal command output.

But artifact-level verification did not pass in the captured samples. The
structured eval JSON is the important evidence here:

- Easy bugfix runs reported `opencode_exit_code=0` and at least one tool turn,
  but `task_success=false` with `buggy_exists=false` and `test_exists=false`.
- Medium scaffold runs reported tool activity, but verification still found no
  expected README/test candidates.
- Hard rigid-prompt runs detected tool-call collapse or no-tool-use conditions;
  one stabilize retry was attempted after literal commands, but it fell back
  because the retry did not produce usable `tool_calls`.

That matters. The public claim should be:

> The proxy repairs an important class of tool-call protocol failures. It is not
> yet proof that stock local-model agent loops reliably complete real tasks.

## Evidence Summary

| Evidence | Result |
| --- | --- |
| Mock JSON-in-content upstream | Rewritten into valid OpenAI `tool_calls` |
| Automated tests | Passing locally |
| Live Ollama `/v1/models` smoke | Proxied real model list with compat markers |
| Live Ollama tool-call smoke | Returned native non-streaming and streaming `tool_calls` through proxy |
| Sanitized JSONL tracing | Wrote request/collapse/replay metadata without raw prompts |
| Real OpenCode compatibility path | Triggered on Gemma 4 E4B MLX runs |
| Drift/collapse logging | Captured useful categories and trace ids |
| Stabilize retry path | Exercised once on a rigid prompt, without recovery |
| Artifact-level task verification | Not yet successful in captured real-harness runs |

Evidence sources used for this summary:

- `docs/history/2026-05-26-gemma4-agentic-testing.md`
- `eval/results/summary_stabilize_soft_2026-05-27.md`
- `eval/results/*.json`
- `eval/logs/proxy_*.log`

The raw logs and full per-run JSON are useful audit material, but they are noisy
and may contain machine-specific paths or prompt text. Do not commit wholesale
log dumps to the public repo unless they are curated, redacted where needed, and
small enough to review. Prefer compact summaries plus selected fixtures that
exercise parser behavior.

## Slightly Positive Ending

The useful result is not that the whole local-agent stack is solved. It is that
the failure became smaller and more legible.

Before the proxy, the problem looked like "local models cannot use tools
reliably." After the proxy, one concrete layer is isolated: local models and
OpenAI-compatible clients often disagree about tool-call shape. That layer can
be repaired, tested, and improved without pretending to solve every agentic
failure at once.

That is a real improvement. The mock and automated tests show protocol repair
works for a class of malformed outputs. The real harness runs show the repaired
protocol can get as far as tool turns and diagnostics, but they also show the
remaining gap: verified artifacts are the standard that matters for coding
agents.

## Next Experiments

1. Continue adding real malformed outputs as parser fixtures.
2. Compare LiteLLM against `local-tool-proxy` on the same tasks.
3. Expand the experimental streaming rewrite path toward true incremental tool-call deltas.
4. Re-run artifact-verified OpenCode tasks after each compatibility improvement.
5. Keep exit-code success separate from task-artifact success.
6. Curate a small public `docs/history/` trail rather than committing full raw
   eval logs.
