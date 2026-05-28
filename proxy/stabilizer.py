"""
proxy/stabilizer.py

Active stabilization logic for NextGrok.

This module has evolved beyond the original conservative v1 design based on real
rigid-prompt data. The retry is now significantly stronger for known failure modes
(especially "model prints shell commands as text").

**Current tactics on stabilize retry (in rough order of strength):**
- tool_choice: "required" at the API level (if the backend honors it)
- Prepended steering message (stronger effect than appending for many local models)
- Category-specific surgical instructions (literal_commands + tool_intent_prose)
- "Analyze failure → adapt strategy → exactly one tool call" reasoning nudge (Haystack-derived)
- Explicit + prescriptive list of available tool names ("use ONLY from this list")
- Optional planner hints (when --planner soft is active)

**Safety / Contract:**
- Only active when `--mode stabilize` is explicitly passed.
- At most one retry per turn by default.
- All interventions are internal and fully logged with trace id.
- We always fall back to the original response on failure.
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

# Strengthened category-specific steering messages for stabilize retries.
# Haystack-inspired lightweight nudges (from their Gemma 4 E4B agent examples):
# - Explicit "analyze what went wrong → adapt strategy" language
# - Strong "think step by step → exactly one tool call" discipline
# - "Use only tools from the exact available list" guardrail
#
# These remain strictly internal, logged, and only affect the single upstream retry
# inside --mode stabilize. They do not rewrite user tasks or create an agent runtime.
LITERAL_COMMANDS_STEERING = (
    "Your previous response printed shell commands or numbered steps as plain text instead of using the tool interface. "
    "Analyze what went wrong in that approach. Adapt your strategy. "
    "Think step by step about only the very first action you need to take right now. "
    "Identify the exact tool name from the available list. "
    "Then emit EXACTLY ONE properly formatted tool call for that first command only. "
    "Do not describe it. Do not continue planning in prose. Output only the tool call."
)

TOOL_INTENT_PROSE_STEERING = (
    "Your previous response described intended actions in prose instead of using the tool interface. "
    "Analyze what went wrong. Adapt your strategy. "
    "Think step by step about the single next concrete step required by the task. "
    "Use only a tool name that exists in the available list. "
    "Emit exactly one valid tool call for that step. "
    "If the task is truly complete, give a final answer instead. Otherwise use a tool."
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
    *,
    prepend_steering: bool = False,
    tool_choice: Optional[str] = None,
    available_tool_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Build the payload for the stabilization retry.

    Improvements (driven by real rigid-prompt data):
    - Support prepending the steering message (stronger effect on some models).
    - Support injecting tool_choice (e.g. "required") at the API level.
    - Optionally include a compact list of available tools in the message.
    """
    payload = dict(original_payload)
    payload["stream"] = False

    if tool_choice:
        payload["tool_choice"] = tool_choice

    messages = list(payload.get("messages", []))
    role = "system"

    steering = steering_message or DEFAULT_STEERING_MESSAGE

    # Make the steering message prescriptive about tool names (Haystack "SearchableToolset" lesson:
    # small models do better when the catalog is explicit and they are told not to invent names).
    if available_tool_names:
        tool_list = ", ".join(available_tool_names[:8])
        if tool_list:
            steering = (
                f"{steering} "
                f"You may ONLY call tools from this exact list: [{tool_list}]. "
                "Never invent tool names. If no tool applies, give a final answer."
            )

    steering_msg = {"role": role, "content": steering}

    if prepend_steering:
        # Stronger for many local models (Gemma 4 E4B observed behavior):
        # Put the steering instruction early in the transcript.
        # When this is a collapse recovery, the steering text already contains the
        # "think step by step → exactly one tool call" nudge + available-tools guardrail.
        messages = [steering_msg] + messages
    else:
        messages.append(steering_msg)

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
