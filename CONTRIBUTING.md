# Contributing

Thanks for taking a look at `local-tool-proxy`.

This project is intentionally narrow: it repairs local-model tool-call protocol
compatibility. It should not become a hidden agent runtime, a tool executor, or
a general model router.

The project is still experimental. Contributions are most useful when they make
behavior easier to inspect, test, or explain.

## Development Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check .
```

Use a virtual environment if you do not want editable install dependencies in
your user Python environment.

## Useful Commands

Run the mock upstream rewrite integration test:

```bash
python3 -m pytest tests/test_integration_mock.py
```

Start the proxy against Ollama:

```bash
local-tool-proxy \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx
```

Run a real harness evaluation, if OpenCode and Ollama are installed locally:

```bash
python3 -m eval.run_real_opencode_eval \
  --difficulty easy \
  --mode compat \
  --planner disabled \
  --timeout 180
```

## What Makes A Good Contribution

- A new parser should include real or realistic malformed model output.
- A bug fix should include a regression test.
- Model-specific behavior should be gated by configuration or model matching.
- Experimental behavior should be opt-in.
- Logging should be useful without exposing raw prompts by default.
- Public docs should be honest about limitations.
- Compatibility changes should preserve ordinary pass-through behavior for
  requests that do not need repair.

## Parser Changes

Parser changes belong primarily in `proxy/rewriters.py`.

When adding a new input format, include tests for:

- The happy path.
- A malformed version that should fail safely.
- Similar prose that should not be parsed as a tool call.
- Known-tool-name behavior, if relevant.

## Pull Request Checklist

- `python3 -m pytest` passes.
- `python3 -m ruff check .` passes.
- New behavior is documented.
- Experimental behavior is behind a flag.
- No raw private prompts, local paths, tokens, or machine-specific logs are committed.
- The proxy still never executes tools.

## Scope Boundaries

Please read [docs/constitution.md](docs/constitution.md) before proposing major
changes. The short version: compatibility first, transparent by default, no
hidden autonomy.

If a change would require the proxy to execute tools, inspect a workspace, own a
task plan, or decide that a coding task is complete, it is outside the current
project boundary.
