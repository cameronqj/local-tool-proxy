"""
proxy/collapse.py

Pure classifier for assistant responses in tool-using conversations.

Part of NextGrok instrumentation (Phase 0).

Classifies why a model stopped using tools (or didn't) so we can
detect "Agent Loop Collapse" vs. legitimate final answers.
"""

from __future__ import annotations
import re
from typing import Literal, Optional, Dict, Any

CollapseCategory = Literal[
    "tool_calls",
    "final_answer",
    "tool_intent_prose",
    "literal_commands",
    "chat",
    "unclear",
]

# High-signal patterns for literal command output (common when small models abandon tools)
LITERAL_COMMAND_PREFIXES = (
    "git ", "mkdir ", "cd ", "python ", "pytest", "uvicorn ", "npm ", "yarn ",
    "echo ", "cat ", "touch ", "rm ", "mv ", "cp ", "ls ", "curl ", "wget ",
    "docker ", "make ", "cargo ", "go run", "go build",
)

# Patterns that strongly suggest the model is describing an action it intends to take
# but is not actually calling a tool.
TOOL_INTENT_PHRASES = (
    "i will now", "next i", "i need to", "i should", "let me", "i'll ",
    "first ", "then ", "after that", "now i will", "to do this i",
    "step ", "1. ", "2. ", "3. ",  # when combined with action verbs
)

ACTION_VERBS = (
    "create", "write", "edit", "modify", "run", "execute", "start",
    "implement", "add", "fix", "update", "install", "commit", "push",
    "test", "build", "deploy",
)


def _looks_like_literal_command(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    # Starts with a common command
    for prefix in LITERAL_COMMAND_PREFIXES:
        if t.startswith(prefix):
            return True
    # Common in rigid prompts: numbered lines that are just shell commands
    if re.match(r"^\d+[\.\)]\s*(git |mkdir |cd |python |pytest|echo |cat |ls )", t):
        return True
    return False


def _contains_tool_intent_prose(text: str) -> bool:
    t = text.lower()
    has_intent = any(phrase in t for phrase in TOOL_INTENT_PHRASES)
    has_action = any(verb in t for verb in ACTION_VERBS)
    # "I will now create the file..." style language without an actual tool call
    return has_intent and has_action


def classify_assistant_response(
    message: Optional[Dict[str, Any]] = None,
    had_tools_in_request: bool = False,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> CollapseCategory:
    """
    Classify a single assistant response.

    Returns one of the CollapseCategory values.

    This is intentionally heuristic and pure (no I/O, no side effects).
    It is the foundation for observe + stabilize modes.
    """
    if message is None:
        message = {}

    # Prefer explicit tool_calls if present
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return "tool_calls"

    # Extract content
    if content is None:
        content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content or "")

    text = content.strip()
    text_lower = text.lower()

    finish = (finish_reason or "").lower()

    # If the model explicitly said it was done and there were no tools requested in this turn
    if not had_tools_in_request:
        if finish == "stop" and len(text) > 20:
            # Could be chat or a final answer in a non-tool context
            return "chat"
        return "unclear"

    # We are in a tools-present context

    # Strongest positive signal: model actually produced tool_calls (already handled above)

    # finish_reason tool_calls with no actual calls is suspicious but rare
    if finish == "tool_calls":
        return "tool_calls"  # even if empty, the model tried

    # Empty or very short response after tools request
    if not text or len(text) < 5:
        return "unclear"

    # Literal commands printed as text (the classic small-model collapse on rigid prompts)
    if _looks_like_literal_command(text):
        return "literal_commands"

    # Clear "I am about to do X" prose without actually calling a tool
    if _contains_tool_intent_prose(text):
        return "tool_intent_prose"

    # Looks like a normal chat / explanatory response
    if any(word in text_lower for word in ("hello", "sure", "i think", "perhaps", "maybe", "let me explain")):
        return "chat"

    # If it ends with something that sounds like task completion without tool use
    completion_signals = ("done", "complete", "finished", "all set", "task is complete", "ready to run")
    if any(sig in text_lower for sig in completion_signals) and len(text) < 400:
        return "final_answer"

    # Default when we can't be confident
    return "unclear"


def classify_from_openai_response(
    response: Dict[str, Any],
    had_tools_in_request: bool = False,
) -> CollapseCategory:
    """Convenience wrapper for a full OpenAI-style chat completion response."""
    choices = response.get("choices", []) or []
    if not choices:
        return "unclear"
    msg = choices[0].get("message", {}) or {}
    finish = choices[0].get("finish_reason")
    return classify_assistant_response(
        message=msg,
        had_tools_in_request=had_tools_in_request,
        finish_reason=finish,
    )


def get_collapse_signals(
    category: CollapseCategory,
    had_tools_in_request: bool,
) -> list[str]:
    """Return a list of human-readable signals for logging/traces."""
    signals = []
    if category == "literal_commands":
        signals.append("model_printed_literal_commands")
    elif category == "tool_intent_prose":
        signals.append("model_described_action_without_calling_tool")
    elif category == "final_answer" and had_tools_in_request:
        signals.append("premature_final_answer_while_tools_available")
    elif category == "chat" and had_tools_in_request:
        signals.append("chat_response_instead_of_tool_use")
    elif category == "unclear" and had_tools_in_request:
        signals.append("no_tool_use_in_tools_context")
    return signals
