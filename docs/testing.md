# Testing

## Unit And Integration Tests

Run:

```bash
python3 -m pytest
```

The current test suite covers:

- Parser behavior in `proxy/rewriters.py`.
- Collapse classification.
- Stabilization retry payloads.
- A mock upstream integration path that verifies JSON-in-content is rewritten to
  OpenAI `tool_calls`.

The tests are meant to protect protocol behavior. They are not a benchmark of
model quality.

## Linting

Run:

```bash
python3 -m ruff check .
```

Before opening a PR, run both pytest and Ruff from the repository root.

## Live Harness Evaluation

Live harness evaluation requires local dependencies such as Ollama, OpenCode,
and a compatible model.

Example:

```bash
python3 -m eval.run_real_opencode_eval \
  --difficulty easy \
  --mode compat \
  --planner disabled \
  --timeout 180
```

The evaluation runner records:

- Duration.
- OpenCode exit code.
- Proxy collapse categories.
- Stabilization attempts.
- Tool-turn signals.
- Artifact-level task verification.

Artifact verification matters because a harness can exit with code `0` without
actually completing the requested coding task.

Treat live harness runs as evidence, not as a deterministic test suite. Local
model output, hardware, prompts, and harness versions can all affect results.

## Raw Logs

Raw eval logs and JSON result files are intentionally ignored by default. Promote
important findings into `docs/experiment-log.md` rather than committing large or
private local traces.

When adding a new fixture from a real model failure, remove private paths,
credentials, and project-specific source before committing it.
