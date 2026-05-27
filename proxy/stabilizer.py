"""
proxy/stabilizer.py

Minimal active stabilization logic for NextGrok (Phase 2).

**Safety / Contract (critical):**
- Only active when `--mode stabilize` is explicitly passed.
- Only triggers on `tool_intent_prose` and `literal_commands` (conservative).
- At most one retry per turn by default (`--stabilize-max-retries`).
- The steering message is added as an *internal* message only.
- The original user task/messages are **never** modified.
- Every intervention is logged with the trace id.
- If the retry does not help, we always fall back to the original response.

This is experimental middleware. It changes model behavior on retries.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

# Conservative trigger set for Stabilize v1 (per nextgrok.prompt)
TRIGGER_CATEGORIES = {"tool_intent_prose", "literal_commands"}

# Default steering message (internal only, never shown to harness user)
DEFAULT_STEERING_MESSAGE = (
    "The previous response described an action but did not use a tool. "
    "Use one of the available tools now, or give a final answer only if the task "
    "is genuinely complete."
)


def should_attempt_stabilize(
    category: str,
    mode: str,
    had_tools: bool = True,
) -> bool:
    """
    Decide whether to attempt a stabilization retry.

    Conservative rules for v1:
    - Only in "stabilize" mode
    - Only on specific collapse categories
    - Only when tools were requested in this turn
    """
    if mode != "stabilize":
        return False
    if not had_tools:
        return False
    return category in TRIGGER_CATEGORIES


def build_retry_payload(
    original_payload: Dict[str, Any],
    steering_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the payload for the stabilization retry.

    Rules (from nextgrok.prompt Retry Contract):
    - Start from the original request body.
    - Append ONE extra internal message (steering).
    - Do NOT modify the original user message(s).
    - Prefer "system" role for the steering message (widely supported).
    - Force stream=false (we are already in the non-stream compat path).
    """
    payload = dict(original_payload)  # shallow copy is fine here
    payload["stream"] = False

    messages = list(payload.get("messages", []))
    role = "system"

    steering = steering_message or DEFAULT_STEERING_MESSAGE

    # Append the steering message as an additional system message at the end.
    # This is the safest place that doesn't alter user intent or previous assistant turns.
    messages.append({"role": role, "content": steering})
    payload["messages"] = messages

    return payload


def should_use_retry_response(
    original_response: Dict[str, Any],
    retry_response: Dict[str, Any],
    return_on_failed: str = "original",
) -> Tuple[bool, str]:
    """
    Decide whether to return the retry response or fall back.

    Returns (use_retry, reason)

    Current v1 policy:
    - If the retry produced valid tool_calls (after our normal rewriter would see it),
      we prefer the retry.
    - Otherwise fall back to original (per spec default).
    """
    def _has_tool_calls(resp: Dict[str, Any]) -> bool:
        choices = resp.get("choices", []) or []
        if not choices:
            return False
        msg = choices[0].get("message", {}) or {}
        return bool(msg.get("tool_calls"))

    retry_has_tools = _has_tool_calls(retry_response)

    if retry_has_tools:
        return True, "retry_produced_tool_calls"

    if return_on_failed == "retry":
        return True, "forced_retry"

    return False, "retry_failed_fallback_to_original"
