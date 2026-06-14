# Real Harness Evaluation Summary — stabilize + planner soft

**Date**: 2026-05-27
**Proxy config**: `--mode stabilize --planner soft --stabilize-max-retries 1 --compat-models gemma4:e4b-mlx`
**Model**: gemma4:e4b-mlx (via Ollama MLX)
**Harness**: stock `opencode run -m small-local/gemma4:e4b-mlx --pure`

## Runs Performed

| Difficulty | Prompt | Duration | Exit Code | Tool Turns (observed) | Collapse Events | Stabilize Attempts | Notes |
|------------|--------|----------|-----------|-----------------------|-----------------|---------------------|-------|
| Medium     | task-01-scaffold-cli | 46.3s   | 0         | 2+                    | 0 (in captured window) | 0 | Completed quickly. Rigid prompt detector fired. Drift tracking active. |
| Easy       | task-02-bugfix       | 45.8s   | 0         | 1+                    | 0                 | 0 | Completed. No collapse that triggered stabilize in this short run. |

## Key Observations from Proxy Logs (stabilize + soft active)

- Multiple `COMPAT TOOL PATH` entries with `mode=stabilize`.
- Drift scoring was active (`drift: {tool_streak, turns_since_last_tool, ...}`).
- No `STABILIZE ATTEMPT` logged in these particular short runs — the model did not emit the specific collapse categories (`tool_intent_prose` / `literal_commands`) that trigger the retry logic.
- "RIGID STRUCTURED PROMPT" detector fired even on the medium task (due to structured language in the prompt).

## Metrics Collection

The `eval/run_real_opencode_eval.py` script now supports:
- `--external-proxy` + `--proxy-log-file` for clean metric extraction from live proxy.
- Structured JSON output per run with collapse / stabilize / drift counts.

## Next Recommended Steps

1. Run the same prompts with `--mode compat` as baseline (apples-to-apples).
2. Run longer / more complex tasks where collapse is more likely.
3. Capture full proxy stdout to a file during eval runs for better post-processing.
4. Add success criteria parsing (did the task actually complete? did tests pass?).

This gives us the first real-harness data points with the full NextGrok stack enabled.
