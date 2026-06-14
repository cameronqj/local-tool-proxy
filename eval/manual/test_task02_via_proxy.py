#!/usr/bin/env python3
"""
More realistic simulation of stock OpenCode doing Task 02 via the proxy.

This script drives a tool-calling loop (similar to what OpenCode does)
against the proxy, trying to complete the clean "bugfix" task:
- Create buggy.py with the known-bad longest-word function
- Fix the bugs (punctuation stripping + tie handling)
- Write test_buggy.py with >=3 pytest cases that would have failed before
- Run pytest and confirm everything passes

Usage (with a local Ollama + gemma4:e4b-mlx running):
    python3 eval/manual/test_task02_via_proxy.py

It will use the proxy at http://localhost:9000 by default (start it first with ./proxy/start.sh).

On this machine it can run against a mock for validation of the rewrite path.
"""

import os
import json
import tempfile
import shutil
import subprocess
import sys
from contextlib import contextmanager
from openai import OpenAI

PROXY_BASE = os.environ.get("PROXY_BASE", "http://localhost:9000/v1")
MODEL = os.environ.get("MODEL", "gemma4:e4b-mlx")

client = OpenAI(base_url=PROXY_BASE, api_key="not-needed")

# The exact task from tasks/task-02-bugfix.md
TASK_PROMPT = '''TASK: Fix a small but real bug in provided code and add a test.

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

Work step by step using tools (read_file, write_file, run_command, etc.).
'''

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the current working directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if necessary",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command (e.g. pytest, python -c '...') and return its output",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

@contextmanager
def isolated_workspace():
    """Run the entire test in a fresh temp directory + venv so it's self-contained."""
    tmpdir = tempfile.mkdtemp(prefix="proxy_task02_test_")
    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        # Create a clean venv inside the workspace
        venv_dir = os.path.join(tmpdir, ".venv")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        venv_python = os.path.join(venv_dir, "bin", "python") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "python.exe")

        # Install pytest + any other test deps
        subprocess.check_call([venv_python, "-m", "pip", "install", "--quiet", "pytest"])

        yield tmpdir, venv_python
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print(f"Testing proxy at {PROXY_BASE} with model {MODEL}")
    print("Attempting to complete Task 02 (bugfix + tests) via tool calls in an isolated workspace...\n")

    with isolated_workspace() as (tmpdir, venv_python):
        messages = [
            {
                "role": "system",
                "content": "You are a careful Python developer. Use tools to read, write, and run commands. When the task is complete, stop.",
            },
            {"role": "user", "content": TASK_PROMPT},
        ]

        max_turns = 12
        for turn in range(1, max_turns + 1):
            print(f"--- Turn {turn} ---")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=600,
            )

            msg = resp.choices[0].message
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

            if msg.tool_calls:
                print(f"Tool calls: {[t.function.name for t in msg.tool_calls]}")
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"  -> {name}({args})")

                    # Execute the tool inside the isolated venv workspace
                    if name == "read_file":
                        path = args["path"]
                        try:
                            with open(path) as f:
                                content = f.read()
                        except FileNotFoundError:
                            content = f"[File not found: {path}]"
                        result = content
                    elif name == "write_file":
                        with open(args["path"], "w") as f:
                            f.write(args["content"])
                        result = f"Wrote {len(args['content'])} bytes to {args['path']}"
                    elif name == "run_command":
                        try:
                            out = subprocess.check_output(args["command"], shell=True, text=True, stderr=subprocess.STDOUT, executable=venv_python)
                        except subprocess.CalledProcessError as e:
                            out = e.output
                        result = out.strip()[:2000]
                    else:
                        result = f"[Unknown tool: {name}]"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    })
                    print(f"     result: {result[:120]}...")
            else:
                print("Assistant message (no tool calls):")
                print((msg.content or "")[:300])
                if "success" in (msg.content or "").lower() or "all tests passed" in (msg.content or "").lower():
                    print("\n✅ Task appears complete according to the model.")
                    break

        print("\n--- Final state of files (if any were created) ---")
        for f in ["buggy.py", "test_buggy.py"]:
            if os.path.exists(f):
                print(f"\n=== {f} ===")
                with open(f) as fh:
                    print(fh.read()[:800])

        print("\nDone. On real hardware with a capable model this often completes the task when the proxy rewrite is active.")


if __name__ == "__main__":
    main()
