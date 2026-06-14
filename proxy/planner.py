"""
proxy/planner.py

Lightweight "Soft Planner" for local-tool-proxy stabilize experiments.

**Safety / Contract (critical):**
- Only has any effect when `--mode stabilize --planner soft` is used.
- Agenda and hints are **never** injected into the original user messages.
- Only used to slightly enrich the internal steering message on stabilize retries.
- Fully disabled by default (`--planner disabled`).
- Must be possible to exactly replay any run with `--mode compat`.

Per the spec:
- Never becomes the primary task decomposer (the harness owns the plan).
- Output lives in trace metadata / is used only for recovery guidance.
- Only influences behavior inside stabilize-mode recovery paths.

Phase 3 v1 scope: very conservative (heuristic extraction only).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

PLANNER_MODES = {"disabled", "observe", "soft"}


def extract_milestones(
    first_user_message: str,
    tool_names: List[str],
    max_milestones: int = 6,
) -> List[str]:
    """
    Very lightweight heuristic agenda extraction for Phase 3 v1.

    Not a full planner — just pulls obvious high-level steps from the user's request.
    This is intentionally simple so we can later replace it with a model call if desired.
    """
    if not first_user_message:
        return []

    text = first_user_message.lower()

    # Very basic sentence / step splitting
    sentences = re.split(r'[.!?\n]+', first_user_message)
    candidates = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Look for action-oriented language
        if any(kw in s.lower() for kw in ["create", "fix", "add", "write", "implement", "build", "make", "test", "run"]):
            candidates.append(s[:120])

    # Also surface known tools as possible milestones
    for tool in tool_names[:3]:
        candidates.append(f"Use {tool} as needed")

    # Dedup and cap
    seen = set()
    milestones = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            milestones.append(c)
        if len(milestones) >= max_milestones:
            break

    return milestones or ["Complete the requested task using the available tools"]


def build_planner_hint(agenda: List[str], observed_progress: Optional[Dict[str, Any]] = None) -> str:
    """
    Produce a short, high-signal hint that can be injected into a recovery steering message
    when planner=soft.
    """
    if not agenda:
        return ""

    hint = "High-level milestones observed so far: " + " | ".join(agenda[:4])

    if observed_progress:
        files = observed_progress.get("files_edited", 0)
        tests = observed_progress.get("tests_passing", False)
        if files or tests:
            hint += f" (progress: {files} files changed, tests passing: {tests})"

    return hint


def should_use_planner_for_recovery(planner_mode: str, stabilize_mode: bool) -> bool:
    """Planner only affects recovery inside stabilize mode."""
    return planner_mode in {"observe", "soft"} and stabilize_mode
