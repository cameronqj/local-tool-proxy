# Deep Dive: Why Small Models Stop Using Tools Under Rigid Constraints

This note records one important negative result from the `local-tool-proxy`
experiments: protocol repair helps only while the model is still expressing tool
intent. It does not make a small model complete a long, rigid task after the
model has abandoned tool use entirely.

## Observed Pattern

On highly constrained prompts (especially the original 8-rule Tic-Tac-Toe prompt), `gemma4:e4b-mlx` frequently stops emitting tool calls and instead prints the intended shell/git commands directly as text inside the numbered steps.

Example output fragments:
```
23. git add .
24. git commit -m "Implement FastAPI backend and game logic"
...
57. echo "<h1>Tic-Tac-Toe</h1>" > templates/index.html
```

No `<tool_code>`, no JSON, no `tool_calls` structure.

## Root Causes (Ranked by Evidence)

1. **Extreme Output Format Pressure + "Show Every Command" Rule**
   - The prompt simultaneously demands:
     - Perfect numbered step format (Rule 1)
     - Exactly 5 specific commit messages in order (Rule 3)
     - "Show every command you execute" (Rule 2)
     - "All output must strictly follow the rules" + "no extra commentary"
   - Satisfying "show the command" is easiest by literally printing the command text.

2. **Long-Horizon Task + Many Micro-Constraints**
   - The model must track 8 overlapping rules while executing a multi-phase coding task.
   - Tool calling requires maintaining two mental models (the task + the exact tool schema).
   - Under cognitive load, the model falls back to its strongest pre-training behavior: describing actions in natural language / literal commands.

3. **Weak Reinforcement for Staying in Tool Mode**
   - The prompt rewards producing the exact strings and numbered format more strongly than it penalizes abandoning the tool interface.
   - Once the model finds a way to produce the visible required output (the commit messages + numbered steps) by printing commands, that strategy is reinforced.

4. **Model Scale Limitations**
   - 4B-class models have limited "working memory" for complex instruction following + tool use over many turns.
   - When the prompt contains both strict formatting rules and a requirement to use tools for everything, the formatting rules win because they are more explicit and immediately verifiable in the output.

## Evidence from Our Runs

- **Test A** (strong "only output tool calls, no extra text"): Model still attempted creative tool syntax (`<tool_code>`, etc.).
- **Original Tic-Tac-Toe** (many overlapping micro-rules + exact strings): Model largely abandoned tools and printed commands as text.
- When we reduced rigidity (Test B), tool use was more persistent but still imperfect on the STEP format.

## Implications for local-tool-proxy

- **Rewriter improvements (Phase 2)** help when the model is *still trying* to use tools but uses weird syntax. They cannot help when the model has completely exited tool-calling mode.
- This is one of the hardest remaining failure modes for small-model agentic work on rigid tasks.
- The proxy can mitigate some cases (by making tool use more reliable when the model stays in the format), but cannot solve the root "model gives up on tools" problem.

That distinction is central to the public story. The positive evidence is the
mock and automated repair path: malformed tool intent can be normalized into
OpenAI-compatible `tool_calls`. The real-harness evidence is more cautious:
OpenCode runs reached the compatibility path and sometimes produced tool turns,
but artifact checks still failed in the captured samples.

In other words, the project has made the protocol layer smaller and testable. It
has not yet proved reliable end-to-end local coding agents on stock small models.

## Potential Mitigations (Outside Pure Rewriting)

- Stronger tool-use reinforcement in prompts ("You MUST use the provided tools for every action. Printing commands as text is not allowed.")
- Different tool schema design (fewer, broader tools vs many specific ones).
- Post-processing / "rigid mode" that detects long stretches of literal commands and injects a correction.
- Accepting that some ultra-rigid prompts are beyond current small-model capability on this hardware and designing the harness to be more resilient.

The stabilize experiments are a first step in this direction. In one hard run,
the proxy detected literal commands and attempted an internal steering retry,
but the retry still did not produce usable tool calls. That is useful evidence,
not a failure to hide: it shows where protocol repair ends and model/harness
behavior begins.
