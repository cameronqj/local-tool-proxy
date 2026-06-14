#!/usr/bin/env python3
"""
Minimal test that simulates what stock OpenCode does with a small model
through the proxy.

It uses the official OpenAI client against the proxy and tries to get
the model to complete Task 02 (fix the longest-word bug + write tests).

This is the closest we can get to "stock OpenCode via proxy" without
actually running the full OpenCode binary.

Run this while the proxy is running:
    python3 eval/manual/test_with_clean_task.py
"""

import os
import json
from openai import OpenAI

# Point this at your running proxy
PROXY_BASE = os.environ.get("PROXY_BASE", "http://localhost:9000/v1")
MODEL = os.environ.get("MODEL", "gemma4:e4b-mlx")

client = OpenAI(base_url=PROXY_BASE, api_key="not-needed")

TASK_PROMPT = """You are an expert Python developer.

There is a file called buggy.py with this function:

def find_longest_word(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    return max(words, key=len)

It has two bugs:
1. It does not strip punctuation (.,!? etc.)
2. It does not correctly handle ties (should return the first occurrence)

Your job:
- Read the current buggy.py (use the read_file tool if available, otherwise assume the content above)
- Create a fixed version of the function
- Write a test file test_buggy.py with at least 3 pytest tests that would have failed on the original buggy version (include punctuation and tie cases)
- Make sure the tests pass

Work step by step using tools. When you are done, the tests must pass.
"""

def main():
    print(f"Testing proxy at {PROXY_BASE} with model {MODEL}")
    print("Sending a tool-using task similar to what stock OpenCode would do...\n")

    messages = [
        {"role": "system", "content": "You are a helpful coding agent. Use tools to read/write files and run commands."},
        {"role": "user", "content": TASK_PROMPT},
    ]

    # Define some common tools that OpenCode-like agents use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command (e.g. pytest)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=800,
        )
    except Exception as e:
        print(f"Error calling proxy: {e}")
        return

    message = resp.choices[0].message
    print("Response finish_reason:", resp.choices[0].finish_reason)
    print("Has tool_calls:", bool(message.tool_calls))

    if message.tool_calls:
        print("\n✓ SUCCESS: Model produced proper tool_calls through the proxy!")
        for tc in message.tool_calls:
            print(f"  - {tc.function.name}({tc.function.arguments[:80]}...)")
    elif message.content:
        print("\nModel responded with content (no tool call this turn):")
        print(message.content[:400] + "..." if len(message.content) > 400 else message.content)

        # The proxy should have already rewritten this case for non-streaming tool requests.
        print("\n(If this contains JSON tool call syntax, the proxy rewrite didn't trigger on this run.)")
    else:
        print("\nEmpty response from model.")


if __name__ == "__main__":
    main()