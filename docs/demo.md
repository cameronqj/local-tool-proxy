# Demo Transcript

This is a captured run of the self-contained demo. It needs no Ollama, no GPU,
and no model download — it starts an in-process mock "local model" that emits the
broken tool-call shapes small models produce, puts the real proxy in front of it,
and prints the before/after for each case.

Run it yourself:

```bash
make demo
# or
python3 demo.py
```

## What you see

The output below is a real run, not a mock-up. Every line is produced by the
proxy repairing (or deliberately *not* repairing) the mock upstream's output.

```text
local-tool-proxy demo  (mock upstream — no Ollama)

[1] JSON object in content
    model emitted   : {"name": "write_file", "arguments": {"path": "buggy.py", "content": "fixed code here"}}
    harness receives: write_file({"path": "buggy.py", "content": "fixed code here"})
    ✓ repaired into write_file

[2] toolName{...} snippet
    model emitted   : run_terminal_cmd{'command': 'pytest -q'}
    harness receives: run_terminal_cmd({"command": "pytest -q"})
    ✓ repaired into run_terminal_cmd

[3] XML <tool_code> block
    model emitted   : <tool_code>run_command("mkdir multi_tool_test")</tool_code>
    harness receives: run_command({"command": "mkdir multi_tool_test"})
    ✓ repaired into run_command

[4] Plain prose (must NOT be turned into a tool call)
    model emitted   : I will write_file after I inspect the project. First I need to understand the existing tests, then I will decide what to change.
    harness receives: (no tool_calls — left as assistant content)
    ✓ left as prose (precision preserved)

4/4 cases behaved as expected
```

## Reading the result

- **Cases 1–3** are the supported claim: the model expressed real tool intent in
  a malformed shape (JSON in `content`, a `toolName{...}` snippet, an XML block).
  The proxy normalizes each into a proper OpenAI `tool_calls` message that a stock
  harness can consume. The repair never invents arguments — it only restructures
  what the model already emitted.
- **Case 4** is the matching non-claim: the model produced plain prose with no
  tool call. The proxy leaves it untouched. It does **not** fabricate a
  `write_file` call just because the word appears in the text. This precision
  boundary is what separates "repair malformed intent" from "guess intent."

## Diagnostic fields

When you run the proxy with `--trace-file traces/run.jsonl`, each repaired turn
appends a sanitized JSONL event. For case 1 above the `rewrite` event looks like:

```json
{
  "event": "rewrite",
  "trace_id": "ltp-<id>",
  "model": "demo-local",
  "rewrite_kind": "content_to_tool_calls",
  "tool_names": ["write_file"]
}
```

The trace records *that* a repair happened and *which* tool names were recovered.
It intentionally omits raw prompts and raw model output — see
[configuration.md](configuration.md) and the trace-sanitization test in
`tests/test_claim_boundaries.py`. Turns the proxy declines to repair (cases like
4, or unparseable content) are recorded as `collapse` events with a category such
as `tool_intent_prose`, so a non-repair is observable rather than silent.

## How this maps to tests

The demo cases are pinned by deterministic unit and end-to-end tests, so the
behavior shown here is enforced in CI:

- `tests/test_claim_boundaries.py` — runs the real proxy app and asserts cases
  1–4 through `/v1/chat/completions`, plus the trace fields above.
- `tests/test_rewriters.py` — parser-level coverage for each malformed shape.
- `tests/test_integration_mock.py` — full mock-upstream rewrite via the OpenAI SDK.
