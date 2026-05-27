#!/usr/bin/env python3
"""
Dedicated harness for testing rigid, highly-structured prompts
(like the Tic-Tac-Toe "numbered steps + exact commit messages" style)
through the local-tool-proxy + live small model.

This makes it easy to:
- Feed the exact prompt the user has been testing.
- Capture the model's full output.
- Run automated checks against the rules in the prompt.
- See where the model (and proxy) succeed or fail.

Usage examples:

    # Test A (multi-tool stress)
    python3 -m proxy.test_rigid_prompt \
        --prompt-file prompts/rigid_tests/test_A_multi_tool_stress.txt \
        --expected-commits "run_terminal_cmd,write_file" \
        --require-numbered-steps

    # Test B (structured CLI)
    python3 -m proxy.test_rigid_prompt \
        --prompt-file prompts/rigid_tests/test_B_structured_cli.txt \
        --expected-commits "feat: add CLI scaffold,feat: implement core logic,docs: add usage" \
        --require-numbered-steps

    # Old Tic-Tac-Toe prompt (still works)
    python3 -m proxy.test_rigid_prompt --prompt-file prompts/rigid_tictactoe_example.txt
"""

import argparse
import json
import re
import sys
from openai import OpenAI

def load_prompt(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

def check_numbered_steps_only(text: str) -> tuple[bool, str]:
    """Rule 1 style check: does the output consist almost entirely of numbered steps?"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    numbered = [l for l in lines if re.match(r'^\d+[\.\)]\s', l)]
    if not lines:
        return False, "No output"
    ratio = len(numbered) / len(lines)
    if ratio > 0.85:
        return True, f"Good ({ratio:.0%} lines are numbered)"
    return False, f"Poor ({ratio:.0%} lines are numbered). Leaked commentary."

def check_exact_commits(text: str, required: list[str]) -> tuple[bool, list[str]]:
    """Check whether the model mentioned the exact required commit messages."""
    found = []
    missing = []
    for msg in required:
        if msg in text:
            found.append(msg)
        else:
            missing.append(msg)
    return len(missing) == 0, missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, help="Path to the rigid prompt file")
    parser.add_argument("--model", default="gemma4:e4b-mlx")
    parser.add_argument("--proxy", default="http://localhost:9000/v1")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument(
        "--expected-commits",
        default="",
        help="Comma-separated list of exact commit messages that must appear (for Test A/B etc.)"
    )
    parser.add_argument(
        "--require-numbered-steps",
        action="store_true",
        help="Fail if the output is not mostly numbered steps (Rule 1 style)"
    )
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)
    client = OpenAI(base_url=args.proxy, api_key="not-needed")

    print(f"Running rigid prompt against {args.model} via proxy at {args.proxy}...")
    print("=" * 60)

    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=args.max_tokens,
    )

    output = resp.choices[0].message.content or ""
    print("\n=== MODEL OUTPUT ===\n")
    print(output)
    print("\n=== ANALYSIS ===\n")

    # Configurable checks
    if args.expected_commits:
        required_commits = [m.strip() for m in args.expected_commits.split(",") if m.strip()]
    else:
        # Backward compat default (old Tic-Tac-Toe prompt)
        required_commits = [
            "Initial project setup with README and dependencies",
            "Implement FastAPI backend and game logic",
            "Add Jinja2 templates and frontend",
            "Add session management and win detection",
            "Polish styling and add play again functionality",
        ]

    if args.require_numbered_steps:
        numbered_ok, numbered_msg = check_numbered_steps_only(output)
        print(f"Numbered-steps discipline: {'PASS' if numbered_ok else 'FAIL'} — {numbered_msg}")
    else:
        print("Numbered-steps check: SKIPPED (use --require-numbered-steps if desired)")

    if required_commits:
        commits_ok, missing_commits = check_exact_commits(output, required_commits)
        print(f"Exact commit messages followed: {'PASS' if commits_ok else 'FAIL'}")
        if missing_commits:
            print("  Missing commits:")
            for m in missing_commits:
                print(f"    - {m}")
    else:
        print("Commit message check: SKIPPED (no --expected-commits provided)")

    print("\nRaw token usage:", getattr(resp, 'usage', 'n/a'))

if __name__ == "__main__":
    main()