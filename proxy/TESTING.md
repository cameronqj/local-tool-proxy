# Testing the Proxy with Stock OpenCode (End Goal)

This document describes exactly how to test whether stock, unmodified OpenCode can now successfully use small local models (especially `gemma4:e4b-mlx`) via the proxy for real agentic tasks.

## Success Criteria

Stock OpenCode (no patches, no custom builds) pointed at the proxy should be able to:

1. Discover and select the model (no more "ollama-cloud" ProviderModelNotFoundError).
2. Perform multi-turn tool calling on a clean task.
3. Complete one of our verifiable tasks, e.g. **Task 02** (fix `buggy.py` + write passing `test_buggy.py`).

When the proxy's rewrite logic triggers, you will see clear logs like:
```
REWRITING tool call for gemma4:e4b-mlx: detected JSON in content → proper tool_calls
```

## 1. One-time Setup on Your Machine (the 24 GB M4 Air)

```bash
cd ~/code/sandbox/localmodels   # or wherever this repo lives

# Install proxy deps (lightweight)
python3 -m pip install fastapi uvicorn httpx pydantic

# Make sure your models are available
ollama list | grep -E 'gemma4:e4b-mlx|gemma4:e2b-mlx|gpt-oss:20b'
```

## 2. Start the Proxy (in its own terminal)

```bash
cd ~/code/sandbox/localmodels

python3 -m proxy.server \
  --port 9000 \
  --ollama-base http://localhost:11434/v1 \
  --compat-models gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b
```

Leave this running. Watch the logs — every tool turn for compat models will be logged.

You can check if the proxy is healthy before starting OpenCode:
```bash
curl http://localhost:9000/health
```

## 3. Configure Stock OpenCode

Copy the example:

```bash
cp proxy/examples/opencode-for-proxy.json ~/.config/opencode/opencode.json
# or put it in the project root as opencode.json
```

You can also merge it into your existing config. The key provider is `small-local` pointing at `http://localhost:9000/v1`.

Restart OpenCode after changing the config.

In OpenCode, you should now be able to select models under the "Local Proxy (Gemma 4 + fixes)" provider, including `gemma4:e4b-mlx`.

## 4. The Test Task (Recommended: Task 02 - Bugfix)

Use the clean task definition we created earlier:

```bash
cat tasks/task-02-bugfix.md
```

**Prompt to give stock OpenCode:**

```
Complete the following task exactly as described. Use tools to read, edit, and verify files.

TASK: Fix a small but real bug in provided code and add a test.

You are given a tiny Python file `buggy.py` containing this function:

```python
def find_longest_word(text: str) -> str:
    """Return the longest word in the text. Words are split on whitespace and punctuation should be stripped."""
    words = text.split()
    if not words:
        return ""
    return max(words, key=len)
```

Known issues:
- It does not strip common punctuation (.,!? etc.) so "hello," beats "hello".
- No handling for ties (should return the first occurrence in case of tie).

Your job:
1. Create the file `buggy.py` with the above function (if not already present).
2. Write a clear, minimal fix.
3. Add or update `test_buggy.py` with at least 3 pytest tests that would have failed before the fix (including punctuation and tie cases).
4. Verify: run pytest and show all tests pass.

Success criteria: The tests pass and demonstrate the bug is fixed. Keep changes minimal and correct.
```

Give OpenCode this prompt (or paste the full content of `tasks/task-02-bugfix.md`).

**Do not** add extra instructions like "show every command" or "use numbered steps" — we deliberately stripped those anti-patterns.

## 5. What to Watch For

In the proxy terminal you should see:

- `COMPAT MODE` and `TOOL REQUEST` logs when OpenCode sends a tools request.
- When the model emits JSON in the content field instead of proper `tool_calls`:
  ```
  REWRITING tool call for gemma4:e4b-mlx: detected JSON in content → proper tool_calls
  Synthesized tool_calls: ['write_file', 'run_command', ...]
  ```
- Successful tool execution loops in OpenCode without "I do not have the capability..." or reconnection errors.

## 6. Verification Commands (after OpenCode finishes)

```bash
# From the directory where OpenCode was working
ls -la buggy.py test_buggy.py

python -c "
from buggy import find_longest_word
print(find_longest_word('hello, world!'))
print(find_longest_word('a bb ccc dddd'))
"

python -m pytest test_buggy.py -q --tb=line
```

All tests should pass, and `find_longest_word` should now strip punctuation and handle ties correctly.

## 7. If It Still Fails

- Check proxy logs for rewrite attempts (or lack thereof).
- Try forcing the non-stream path more aggressively (we can add a flag).
- Fall back to the non-streaming test script first:
  ```bash
  PROXY_BASE=http://localhost:9000/v1 MODEL=gemma4:e4b-mlx \
    python3 -m proxy.test_with_clean_task
  ```

## 8. Recommended Validation on Your Real Hardware

After starting the proxy, the best way to test end-to-end is:

```bash
python3 -m proxy.test_task02_via_proxy
```

This script drives a realistic multi-turn tool-calling loop (very similar to what stock OpenCode does) against the proxy, attempting to complete the exact clean Task 02 (bugfix + tests + pytest verification).

It will show you live tool calls, rewrites (if they occur), and the final state of `buggy.py` / `test_buggy.py`.

## 9. Current Known Limitations (as of this build)

- Full incremental streaming tool call deltas are still basic (we force non-stream for tool turns on compat models for reliability — this is often better for small models anyway).
- Complex multi-tool parallel calls or very long argument strings may need more parser work in `rewriters.py`.
- gpt-oss ontology mismatches (container.* etc.) are not yet mapped — focus is on Gemma 4 for now.

## Next Engineering Steps (if this gets us close)

- Improve streaming delta emission.
- Add more parser types from the OpenCode PR #16531 and SmallHarness.
- Tool name normalization for different model families.

---

**Report back** with:
- Did stock OpenCode see the model?
- Did the proxy rewrite any tool calls?
- Did it complete the task (or how far did it get)?
- Paste relevant proxy log lines.

This is the direct measurement of whether the proxy is achieving the original goal.
