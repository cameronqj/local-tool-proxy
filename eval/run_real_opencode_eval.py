#!/usr/bin/env python3
"""
Real harness evaluation runner for NextGrok (stabilize + planner).

Runs stock `opencode` against the proxy in different modes,
with prompts of varying difficulty, and records structured metrics.

The script can either:
- Start the proxy itself with proper logging (recommended for clean runs), or
- Use an already-running proxy via --external-proxy (you must also provide --proxy-log-file for good metrics).

Usage examples:
    # Recommended: let the script manage the proxy + logging
    python -m eval.run_real_opencode_eval --difficulty hard --mode stabilize --planner soft --timeout 300

    # Or manage the proxy yourself
    python -m eval.run_real_opencode_eval --difficulty hard --mode stabilize --planner soft \
        --external-proxy --proxy-log-file proxy_stabilize_soft.log --timeout 300

Metrics collected (per run):
- duration_seconds
- exit_code
- proxy_mode / planner_mode / difficulty
- collapse classification stats:
    - num_classifications
    - num_bad_collapses (only tool_intent_prose + literal_commands)
    - collapse_by_category (dict)
- num_stabilize_attempts
- num_successful_recoveries
- tool_turns (proxy signals)
- task_success (boolean from post-run verification)
- verification_details (per difficulty)

Real success is no longer based solely on opencode exit code.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PROXY_SCRIPT = "python3 -m proxy.server"
OPENCODE_BIN = os.environ.get("OPENCODE_BIN", "opencode")

PROMPTS = {
    "easy": ROOT / "tasks" / "task-02-bugfix.md",
    "medium": ROOT / "tasks" / "task-01-scaffold-cli.md",
    "hard": ROOT / "prompts" / "rigid_tictactoe_example.txt",
}

def load_prompt(difficulty: str) -> str:
    path = PROMPTS[difficulty]
    return path.read_text()

def start_proxy(mode: str, planner: str, port: int = 9000, external: bool = False,
                logs_dir: Path = None, stabilize_max_retries: int = 1,
                debug_log_model_outputs: bool = False) -> tuple[Optional[subprocess.Popen], Optional[Path]]:
    """
    Returns (proxy_process, proxy_log_file_path)
    """
    if external:
        print(f"[eval] Using external proxy on port {port} (mode={mode}, planner={planner})")
        return None, None

    logs_dir = logs_dir or (ROOT / "eval" / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    proxy_log_path = logs_dir / f"proxy_{mode}_{planner}_{ts}.log"

    cmd = [
        "python3", "-m", "proxy.server",
        "--port", str(port),
        "--ollama-base", "http://localhost:11434/v1",
        "--compat-models", "gemma4:e4b-mlx",
        "--mode", mode,
        "--stabilize-max-retries", str(stabilize_max_retries),
    ]
    if debug_log_model_outputs:
        cmd.append("--debug-log-model-outputs")
    if planner != "disabled":
        cmd += ["--planner", planner]

    print(f"[eval] Starting proxy with full logging:")
    print(f"       {' '.join(cmd)}")
    print(f"       Proxy log: {proxy_log_path}")

    with open(proxy_log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # Give the proxy time to start
    time.sleep(4)
    return proc, proxy_log_path

def run_opencode(prompt: str, port: int, workdir: Path, timeout: int = 180) -> Dict[str, Any]:
    """
    Run opencode inside a dedicated workdir so we can later verify actual artifacts.
    """
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(ROOT / "opencode.json")

    cmd = [
        OPENCODE_BIN, "run",
        "-m", "small-local/gemma4:e4b-mlx",
        "--pure",
        prompt,
    ]

    print(f"[eval] Running opencode inside temp workspace (timeout={timeout}s)...")
    print(f"       workdir: {workdir}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,   # Critical: run in isolated temp dir
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
        }
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        return {
            "exit_code": -1,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "duration": duration,
            "timed_out": True,
        }

def parse_proxy_logs(log_text: str) -> Dict[str, Any]:
    """
    Extract improved metrics from proxy logs.

    Fixes the previous bug where every "collapse: category=" line (including
    category=tool_calls) was counted as a bad collapse.
    """
    metrics = {
        "num_classifications": 0,
        "num_bad_collapses": 0,
        "collapse_by_category": {},
        "num_stabilize_attempts": 0,
        "num_successful_recoveries": 0,
        "tool_turns": 0,
        "drift_events": 0,
    }

    BAD_CATEGORIES = {"tool_intent_prose", "literal_commands"}

    collapse_re = re.compile(r"collapse: category=([a-z_]+)")
    stabilize_attempt_re = re.compile(r"STABILIZE ATTEMPT")
    stabilize_success_re = re.compile(r"STABILIZE SUCCESS")
    drift_re = re.compile(r"drift:")
    # Better signals for actual tool-using turns
    tool_turn_re = re.compile(r"(COMPAT TOOL PATH|\"finish_reason\":\s*\"tool_calls\")")

    for line in log_text.splitlines():
        m = collapse_re.search(line)
        if m:
            category = m.group(1)
            metrics["num_classifications"] += 1
            metrics["collapse_by_category"][category] = metrics["collapse_by_category"].get(category, 0) + 1
            if category in BAD_CATEGORIES:
                metrics["num_bad_collapses"] += 1

        if stabilize_attempt_re.search(line):
            metrics["num_stabilize_attempts"] += 1
        if stabilize_success_re.search(line):
            metrics["num_successful_recoveries"] += 1
        if drift_re.search(line):
            metrics["drift_events"] += 1
        if tool_turn_re.search(line):
            metrics["tool_turns"] += 1

    return metrics


def verify_task_success(difficulty: str, workdir: Path) -> Dict[str, Any]:
    """
    Real post-run verification that the task was actually completed.

    This replaces blind reliance on opencode exit_code == 0.
    Runs inside the provided workdir (which should be a fresh temp directory
    for that eval run).
    """
    result = {
        "task_success": False,
        "verification_details": {},
        "artifacts": [],
    }

    if difficulty == "easy":
        buggy = workdir / "buggy.py"
        test_file = workdir / "test_buggy.py"

        result["verification_details"]["buggy_exists"] = buggy.exists()
        result["verification_details"]["test_exists"] = test_file.exists()

        if buggy.exists() and test_file.exists():
            # Try to run pytest
            try:
                pytest_result = subprocess.run(
                    ["python3", "-m", "pytest", str(test_file), "-q", "--tb=no"],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                passed = pytest_result.returncode == 0
                result["verification_details"]["pytest_exit_code"] = pytest_result.returncode
                result["verification_details"]["pytest_output"] = pytest_result.stdout.strip()[-500:]
                result["task_success"] = passed
            except Exception as e:
                result["verification_details"]["pytest_error"] = str(e)

        result["artifacts"] = [str(p.name) for p in workdir.glob("*.py")]

    elif difficulty == "medium":
        # Look for a wordcount-cli style project
        candidates = list(workdir.glob("**/wordcount*")) + list(workdir.glob("**/requirements.txt"))
        result["verification_details"]["found_candidates"] = len(candidates) > 0

        # Check for key files in common locations
        has_readme = any((workdir / "README.md").exists() for _ in [1])
        has_test = bool(list(workdir.glob("**/test_*.py")))
        result["verification_details"]["has_readme"] = has_readme
        result["verification_details"]["has_test"] = has_test

        # Try to find and run the CLI if it looks plausible
        # For now we keep it heuristic; real matrix runs can be stricter
        result["task_success"] = has_readme and has_test and len(candidates) > 0
        result["artifacts"] = [p.relative_to(workdir).as_posix() for p in workdir.rglob("*") if p.is_file()][:15]

    elif difficulty == "hard":
        # Rigid tictactoe: look for git repo + specific commits
        git_dir = workdir / ".git"
        result["verification_details"]["has_git"] = git_dir.exists()

        if git_dir.exists():
            try:
                log = subprocess.run(
                    ["git", "log", "--oneline", "-10"],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                commits = log.stdout.strip().splitlines()
                result["verification_details"]["commit_count"] = len(commits)
                result["verification_details"]["recent_commits"] = commits[:5]

                required_msgs = [
                    "Initial project setup with README and dependencies",
                    "Implement FastAPI backend and game logic",
                ]
                found = sum(1 for c in commits for msg in required_msgs if msg in c)
                result["verification_details"]["required_commit_messages_found"] = found >= 1

                # Very basic success signal for hard
                result["task_success"] = len(commits) >= 3 and (workdir / "requirements.txt").exists()
            except Exception as e:
                result["verification_details"]["git_error"] = str(e)

        result["artifacts"] = [p.name for p in workdir.iterdir() if p.is_file()][:10]

    else:
        result["task_success"] = False
        result["verification_details"]["error"] = "Unknown difficulty"

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--mode", choices=["compat", "observe", "stabilize"], default="stabilize")
    parser.add_argument("--planner", choices=["disabled", "observe", "soft"], default="soft")
    parser.add_argument("--timeout", type=int, default=180, help="opencode timeout in seconds")
    parser.add_argument("--proxy-port", type=int, default=9000)
    parser.add_argument("--external-proxy", action="store_true",
                        help="Do not start the proxy (use this if you started it manually with the desired flags)")
    parser.add_argument("--proxy-log-file", type=Path, default=None,
                        help="Path to proxy log file (required for good metrics when using --external-proxy)")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "eval" / "logs",
                        help="Directory for proxy logs when the script starts the proxy itself")
    parser.add_argument("--stabilize-max-retries", type=int, default=1,
                        help="Passed through to the proxy as --stabilize-max-retries")
    parser.add_argument("--debug-log-model-outputs", action="store_true",
                        help="Pass through to the proxy. Logs raw prompts/tool schemas/model outputs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "results")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args.difficulty)
    print(f"\n=== Starting real eval ===")
    print(f"difficulty: {args.difficulty}")
    print(f"mode: {args.mode}")
    print(f"planner: {args.planner}")
    print(f"prompt length: {len(prompt)} chars\n")

    proxy_proc = None
    proxy_log_path: Optional[Path] = None
    proxy_log_content = ""
    opencode_result = {}
    verification = {"task_success": False, "verification_details": {}}

    # Create an isolated temp workspace for this run (key for real verification)
    with tempfile.TemporaryDirectory(prefix=f"opencode_eval_{args.difficulty}_") as tmpdir:
        workdir = Path(tmpdir)

        try:
            proxy_proc, proxy_log_path = start_proxy(
                args.mode,
                args.planner,
                args.proxy_port,
                external=args.external_proxy,
                logs_dir=args.logs_dir,
                stabilize_max_retries=args.stabilize_max_retries,
                debug_log_model_outputs=args.debug_log_model_outputs,
            )

            time.sleep(2)

            # Run inside the fresh temp workspace
            opencode_result = run_opencode(prompt, args.proxy_port, workdir, args.timeout)

            # Post-run verification (the real success signal)
            verification = verify_task_success(args.difficulty, workdir)

            # Collect proxy logs
            if proxy_log_path and proxy_log_path.exists():
                try:
                    proxy_log_content = proxy_log_path.read_text(errors="ignore")
                except Exception:
                    pass
            elif args.proxy_log_file and args.proxy_log_file.exists():
                try:
                    proxy_log_content = args.proxy_log_file.read_text(errors="ignore")
                except Exception:
                    pass

        finally:
            if proxy_proc:
                print("[eval] Stopping proxy...")
                proxy_proc.terminate()
                try:
                    proxy_proc.wait(timeout=5)
                except Exception:
                    proxy_proc.kill()

    # Parse metrics
    proxy_metrics = parse_proxy_logs(proxy_log_content)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "difficulty": args.difficulty,
        "mode": args.mode,
        "planner": args.planner,
        "stabilize_max_retries": args.stabilize_max_retries,
        "duration_seconds": round(opencode_result.get("duration", 0), 2),
        "opencode_exit_code": opencode_result.get("exit_code"),
        "timed_out": opencode_result.get("timed_out", False),
        **proxy_metrics,
        "task_success": verification.get("task_success", False),
        "verification": verification.get("verification_details", {}),
        "proxy_log_file": str(proxy_log_path) if proxy_log_path else str(args.proxy_log_file) if args.proxy_log_file else None,
        "proxy_log_tail": proxy_log_content[-3000:] if proxy_log_content else "",
        "opencode_stdout_tail": opencode_result.get("stdout", "")[-2000:],
        # For diagnostics: save full output so we can inspect OpenCode's actual behavior
        "opencode_stdout_full_file": None,
    }

    # Save result
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"eval_{args.difficulty}_{args.mode}_{args.planner}_{ts}.json"

    # Diagnostic: save full opencode stdout so we can see what the harness actually did/said
    full_stdout_path = None
    if opencode_result.get("stdout"):
        full_stdout_path = args.output_dir / f"opencode_stdout_{args.difficulty}_{args.mode}_{ts}.txt"
        full_stdout_path.write_text(opencode_result["stdout"])
        record["opencode_stdout_full_file"] = str(full_stdout_path)

    if opencode_result.get("stderr"):
        full_stderr_path = args.output_dir / f"opencode_stderr_{args.difficulty}_{args.mode}_{ts}.txt"
        full_stderr_path.write_text(opencode_result["stderr"])
        record["opencode_stderr_full_file"] = str(full_stderr_path)

    out_path.write_text(json.dumps(record, indent=2))

    print("\n=== Eval complete ===")
    clean_record = {k: v for k, v in record.items() if k not in ("proxy_log_tail", "opencode_stdout_tail")}
    print(json.dumps(clean_record, indent=2))

    # Prominently surface the real success signal
    success = record.get("task_success", False)
    print(f"\n>>> REAL TASK SUCCESS: {success}  (exit_code={record.get('opencode_exit_code')})")

    if record.get("proxy_log_file"):
        print(f"\nProxy log: {record['proxy_log_file']}")
        print(f"  (tail with: tail -f {record['proxy_log_file']})")

    print(f"\nFull record written to: {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
