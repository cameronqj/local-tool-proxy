# gptfixes

OpenAI-compatible tool-call repair proxy for Ollama and other local models.

**Goal**: Make small/local models that are already trying to use tools produce wire-format output that ordinary OpenAI-compatible harnesses (OpenCode, Aider, Continue, etc.) can consume.

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run the proxy (pointing at your local Ollama)
gptfixes --port 9000 --ollama-base http://localhost:11434/v1 --compat-models gemma4:e4b-mlx

# Point your client at the proxy
# Example for OpenCode or any OpenAI client:
# base_url = "http://localhost:9000/v1"
```

## Modes

- `compat` (default): Transparent + tool call normalization only.
- `observe`: Adds diagnostics for debugging model behavior.
- `stabilize` (experimental): Limited interventions for agent-loop drift.

## Current Status

This is early development. See `gptfixes.prompt` for the detailed product specification and roadmap.

## Development

```bash
pip install -e ".[dev]"   # (add dev deps later)
python -m pytest
gptfixes --help
```
