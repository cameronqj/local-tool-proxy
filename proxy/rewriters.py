from __future__ import annotations

"""
Rewriters / adapters for small local models.

Goal: Turn the messy but often-correct output of Gemma 4 E*, gpt-oss, etc.
into the clean streaming tool_calls format that normal OpenCode-like harnesses expect.

This module is intentionally small and focused. It is the core repair logic of
the proxy: turning recoverable-but-malformed tool intent into valid tool_calls.

Current status: tested parsers for JSON-in-content, JSON-ish repairs,
toolName{...} snippets, and XML-ish tool blocks. The shapes here were derived
from real failure traces collected while running small local models against
OpenAI-compatible harnesses. Possible future additions include legacy
function_call variants and reasoning_content stripping.
"""

import json
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# === Reason codes ==========================================================
#
# Every repair/abstention decision is labelled with exactly one of these stable
# string codes. They are the contract the proxy exposes about *why* it did or
# did not turn model output into tool_calls. They are intentionally narrow and
# deterministic so tests can pin them and diagnostics can be scrutinized.
#
# Repair codes (the proxy emitted tool_calls):
REASON_VALID_NATIVE = "valid_native_tool_call"          # upstream already returned tool_calls
REASON_REPAIRED_JSON = "repaired_json_content"          # bare JSON object/array in content
REASON_REPAIRED_MARKDOWN = "repaired_markdown_json"     # ```json fenced block in content
REASON_REPAIRED_XML = "repaired_xml_tool_block"         # <tool_code>/<execute_tool>/<tool> block
REASON_REPAIRED_TOOLNAME = "repaired_toolname_snippet"  # toolName{...} / toolName({...}) snippet
REASON_REPAIRED_LEGACY_FUNCTION_CALL = "repaired_legacy_function_call"  # legacy function_call field
#
# Abstention codes (the proxy deliberately did NOT emit tool_calls):
REASON_ABSTAIN_NO_TOOL_INTENT = "abstain_no_tool_intent"                  # prose, no tool structure
REASON_ABSTAIN_MISSING_REQUIRED_ARGS = "abstain_missing_required_args"    # required arg absent
REASON_ABSTAIN_INVALID_ARGUMENTS = "abstain_invalid_arguments"           # required arg empty/null
REASON_ABSTAIN_UNKNOWN_TOOL = "abstain_unknown_tool"                      # name not in declared tools
REASON_ABSTAIN_AMBIGUOUS_MULTIPLE_TOOLS = "abstain_ambiguous_multiple_tools"  # >1 tool, no clear call
REASON_ABSTAIN_MALFORMED_UNRECOVERABLE_JSON = "abstain_malformed_unrecoverable_json"  # tool-ish, unparseable
#
# Pass-through code (nothing to repair, nothing to abstain from):
REASON_PASS_THROUGH_NO_TOOLS_DECLARED = "pass_through_no_tools_declared"  # request declared no tools

REPAIR_REASONS = frozenset({
    REASON_VALID_NATIVE,
    REASON_REPAIRED_JSON,
    REASON_REPAIRED_MARKDOWN,
    REASON_REPAIRED_XML,
    REASON_REPAIRED_TOOLNAME,
    REASON_REPAIRED_LEGACY_FUNCTION_CALL,
})

ABSTAIN_REASONS = frozenset({
    REASON_ABSTAIN_NO_TOOL_INTENT,
    REASON_ABSTAIN_MISSING_REQUIRED_ARGS,
    REASON_ABSTAIN_INVALID_ARGUMENTS,
    REASON_ABSTAIN_UNKNOWN_TOOL,
    REASON_ABSTAIN_AMBIGUOUS_MULTIPLE_TOOLS,
    REASON_ABSTAIN_MALFORMED_UNRECOVERABLE_JSON,
})


def extract_known_tool_names(request_payload: dict) -> list[str]:
    """Extract tool names from an OpenAI-style request for guided parsing."""
    names = []
    for t in request_payload.get("tools", []) or []:
        fn = t.get("function") or t
        if name := fn.get("name"):
            names.append(name)
    return names


def looks_like_tool_call_json(text: str) -> bool:
    """
    Heuristic: does this text blob look like it is trying to express one or more tool calls
    as JSON (the most common failure mode we see with Gemma 4 E4B/E2B)?
    """
    if not text or not text.strip():
        return False
    text = text.strip()

    # Common patterns (expanded from earlier runs + ToolBridge-style observation)
    patterns = [
        r'^\s*\{\s*"name"\s*:\s*".+?"',                    # {"name": "foo", ...
        r'^\s*\[\s*\{\s*"name"\s*:\s*".+?"',               # array of calls
        r'^\s*\{\s*"tool_calls?"\s*:\s*\[',                # {"tool_calls": [...] } wrapper
        r'<tool_use>.*?</tool_use>',                       # some models use tags
        r'```(?:json)?\s*\{.*?"name".*?\}\s*```',          # fenced JSON block
        r'<tool_code>.*?</tool_code>',                     # common small model XML style
        r'<execute_tool\b',                                # <execute_tool ...> (loose match)
        r'\*\*Tool Call:\*\*',                             # **Tool Call:** markdown style
        r'\b\w+\s*\{',                                     # toolName{...}  (ToolBridge-style JSON fallback)
        r'\b\w+\s*\(\s*\{',                                # toolName({...})
    ]
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def _parse_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse various XML-ish tool call formats that small models invent.

    This version is hardened for the exact creative formats observed in rigid prompt runs
    (e.g. <tool_code>run_command("mkdir foo")</tool_code> with quoted arguments).
    """
    patterns = [
        # <tool_code>tool_name("args...")</tool_code> or with single quotes
        r'<tool_code>\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)\s*</tool_code>',
        # <execute_tool>tool_name(args...)</execute_tool>
        r'<execute_tool>\s*(\w+)\s*\((.*?)\)\s*</execute_tool>',
        # <tool><name>xxx</name><arguments>...</arguments></tool>
        r'<tool>\s*<name>\s*(\w+)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</tool>',
        # Loose tags seen in earlier rigid tests
        r'<(run_terminal_cmd|write_file|git_commit)\b[^>]*>(.*?)</\1>',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            name = match.group(1)
            args_str = match.group(2).strip()

            # Try to parse as JSON first (handles the good cases)
            if args_str.startswith('{'):
                try:
                    args = json.loads(repair_jsonish(args_str))
                    return {
                        "id": f"call_{name[:8]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        }
                    }
                except Exception:
                    pass

            # Fallback: treat as a single positional string argument (common in creative output)
            # e.g. <tool_code>run_command("mkdir foo")</tool_code>
            # becomes {"command": "mkdir foo"} when the tool name suggests a command runner
            command_tool_names = {"run_command", "run_terminal_cmd", "shell", "bash", "execute"}
            if name.lower() in command_tool_names:
                # Strip surrounding quotes if present
                val = args_str
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                args = {"command": val}
            else:
                args = {"raw": args_str}

            return {
                "id": f"call_{name[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                }
            }
    return None


def _extract_balanced_json(text: str, start_index: int) -> Optional[str]:
    """Brace-balanced JSON extraction (inspired by ToolBridge jsonFallback)."""
    brace_count = 0
    in_string = False
    escape = False
    started = False
    end_index = -1

    for i in range(start_index, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                brace_count += 1
                started = True
            elif ch == '}':
                brace_count -= 1
                if started and brace_count == 0:
                    end_index = i + 1
                    break
    if end_index != -1:
        return text[start_index:end_index]
    return None


def repair_jsonish(text: str, repairs: list[str] | None = None) -> str:
    """Apply common JSON repairs (single quotes, unquoted keys, trailing commas, fences)."""
    if not text:
        return text
    repairs = repairs or ["strip_fences", "single_quotes", "unquoted_keys", "trailing_commas"]

    cleaned = text.strip()
    if "strip_fences" in repairs:
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    if "single_quotes" in repairs:
        cleaned = re.sub(r"'", '"', cleaned)
    if "unquoted_keys" in repairs:
        cleaned = re.sub(r'(\w+)\s*:', r'"\1":', cleaned)
    if "trailing_commas" in repairs:
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    return cleaned


def parse_json_content(text: str, known_tool_names: list[str] | None = None) -> list[dict]:
    """Parse JSON or JSON-like tool calls from content."""
    if not text or not looks_like_tool_call_json(text):
        return []

    cleaned = repair_jsonish(text)
    calls = []

    try:
        data = json.loads(cleaned)

        # Unwrap common tool call wrappers: {"tool_calls": [...] } or {"tool_call": [...]}
        if isinstance(data, dict):
            if "tool_calls" in data:
                items = data["tool_calls"] if isinstance(data["tool_calls"], list) else [data["tool_calls"]]
            elif "tool_call" in data:
                items = data["tool_call"] if isinstance(data["tool_call"], list) else [data["tool_call"]]
            else:
                items = [data]
        else:
            items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Handle both flat and nested function shapes
            func = item.get("function") or item
            name = func.get("name") or item.get("name") or item.get("tool")
            if not name:
                continue

            # Support multiple argument key names used by different models/harnesses
            raw_args = (
                func.get("arguments")
                or func.get("parameters")
                or func.get("args")
                or func.get("input")
                or item.get("arguments")
                or item.get("parameters")
                or item.get("args")
                or item.get("input")
                or {}
            )

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {"raw": raw_args}
            else:
                args = raw_args

            calls.append({
                "id": f"call_{len(calls)}_{name[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                }
            })
    except Exception:
        # Try the ToolBridge-style name{json} fallback
        for name in (known_tool_names or []):
            pattern = re.compile(re.escape(name) + r'\s*(?:\(\s*)?(\{)', re.IGNORECASE)
            m = pattern.search(text)
            if not m:
                continue
            brace_idx = text.find('{', m.start())
            if brace_idx == -1:
                continue
            json_str = _extract_balanced_json(text, brace_idx)
            if not json_str:
                continue
            try:
                args = json.loads(repair_jsonish(json_str))
                calls.append({
                    "id": f"call_{len(calls)}_{name[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
                break
            except Exception:
                continue

    return calls


def parse_xml_content(text: str, known_tool_names: list[str] | None = None) -> list[dict]:
    """Parse XML-like tool call formats."""
    # Reuse and slightly generalize the existing _parse_xml_tool_call logic
    result = _parse_xml_tool_call(text)
    return [result] if result else []


def parse_tool_call_from_content(
    content: str,
    known_tool_names: list[str] | None = None,
) -> list[dict] | None:
    """
    Best-effort extraction using a clear strategy list from the historical design notes.

    Strategy order:
    1. JSON content (including name{json} fallback when known names are provided)
    2. XML-like content
    (Additional strategies like legacy function_call can be added here)
    """
    if not content:
        return None

    text = content.strip()
    if not looks_like_tool_call_json(text):
        return None

    known = known_tool_names or []

    # Strategy 1: JSON content (most common)
    json_calls = parse_json_content(text, known)
    if json_calls:
        return json_calls

    # Strategy 2: XML-like content
    xml_calls = parse_xml_content(text, known)
    if xml_calls:
        return xml_calls

    return None


# === Schema-aware repair decision layer ===================================
#
# parse_tool_call_from_content() is deliberately permissive: it recovers
# structure. The decision layer below adds the conservative half — it validates
# recovered calls against the *declared* tool schema and refuses to fabricate a
# call the model did not clearly and completely express. Each decision carries a
# stable reason code (see the constants above).


@dataclass
class RepairDecision:
    """Outcome of inspecting one assistant message for tool intent.

    `tool_calls` is non-empty only when the proxy decided to emit a repaired (or
    pass-through native) tool call. On any abstention it is empty and `reason`
    explains why no call was produced.
    """

    reason: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True when the decision results in emitted tool_calls."""
        return bool(self.tool_calls)


def _required_args_for(tools: Optional[list], name: str) -> Optional[List[str]]:
    """Return the declared required-arg names for `name`, or None if the tool
    is not declared in this request. An empty list means the tool exists but
    requires no arguments."""
    for t in tools or []:
        fn = t.get("function") or t
        if fn.get("name") == name:
            params = fn.get("parameters") or {}
            required = params.get("required") or []
            return [r for r in required if isinstance(r, str)]
    return None


def _loads_args(arguments: Any) -> Dict[str, Any]:
    """Coerce a tool call's `arguments` (string or dict) into a dict."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_empty_arg_value(value: Any) -> bool:
    """A required argument is 'invalid' when present but clearly carries no
    value: None, empty/whitespace string, or empty list/dict. Note that 0 and
    False are valid values and are NOT treated as empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _validate_calls(calls: List[Dict[str, Any]], tools: Optional[list]) -> Optional[str]:
    """Validate recovered calls against the declared schema.

    Returns an abstention reason code if any call is unsafe to emit, else None.
    """
    for call in calls:
        name = call.get("function", {}).get("name")
        required = _required_args_for(tools, name)
        if required is None:
            return REASON_ABSTAIN_UNKNOWN_TOOL
        args = _loads_args(call.get("function", {}).get("arguments"))
        if any(r not in args for r in required):
            return REASON_ABSTAIN_MISSING_REQUIRED_ARGS
        if any(_is_empty_arg_value(args.get(r)) for r in required):
            return REASON_ABSTAIN_INVALID_ARGUMENTS
    return None


_XML_TOOL_TAG = re.compile(
    r"<\s*(tool_code|execute_tool|tool|tool_use|run_terminal_cmd|write_file|git_commit)\b",
    re.IGNORECASE,
)


def _classify_repair_kind(text: str) -> str:
    """Infer which malformed shape was repaired, for an honest reason code."""
    t = text.strip()
    if _XML_TOOL_TAG.search(t):
        return REASON_REPAIRED_XML
    if "```" in t:
        return REASON_REPAIRED_MARKDOWN
    if t[:1] in "{[":
        return REASON_REPAIRED_JSON
    return REASON_REPAIRED_TOOLNAME


def _distinct_known_names_in(text: str, known_names: List[str]) -> set:
    """Distinct declared tool names that appear as whole words in `text`."""
    found = set()
    for name in known_names:
        if re.search(r"\b" + re.escape(name) + r"\b", text):
            found.add(name)
    return found


def _call_from_legacy_function_call(function_call: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy OpenAI `function_call` field into a tool_calls entry."""
    name = function_call.get("name") or ""
    raw_args = function_call.get("arguments")
    if isinstance(raw_args, dict):
        arguments = json.dumps(raw_args)
    elif isinstance(raw_args, str):
        arguments = raw_args
    else:
        arguments = "{}"
    return {
        "id": f"call_0_{name[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def decide_repair(content: str, tools: Optional[list] = None) -> RepairDecision:
    """Decide whether/how to repair tool intent found in assistant `content`.

    `tools` is the OpenAI-style tools list from the request (full specs, used
    for schema-aware validation). This is the conservative core: it only emits
    tool_calls when a complete, schema-valid call can be recovered.
    """
    tools = tools or []
    if not tools:
        # With no declared tools we have nothing to validate against, so we
        # never fabricate a call from content.
        return RepairDecision(REASON_PASS_THROUGH_NO_TOOLS_DECLARED)

    known_names = extract_known_tool_names({"tools": tools})
    text = (content or "").strip()

    # No tool-ish structure at all -> the model expressed no tool intent.
    if not text or not looks_like_tool_call_json(text):
        return RepairDecision(REASON_ABSTAIN_NO_TOOL_INTENT)

    calls = parse_tool_call_from_content(text, known_names or None)

    if not calls:
        # Looked tool-ish but we could not recover a call. If several declared
        # tools are named without one clear invocation, treat it as ambiguous;
        # otherwise the structure is simply too malformed to recover safely.
        if len(_distinct_known_names_in(text, known_names)) >= 2:
            return RepairDecision(REASON_ABSTAIN_AMBIGUOUS_MULTIPLE_TOOLS)
        return RepairDecision(REASON_ABSTAIN_MALFORMED_UNRECOVERABLE_JSON)

    bad = _validate_calls(calls, tools)
    if bad:
        return RepairDecision(bad)

    return RepairDecision(_classify_repair_kind(text), calls)


def decide_message_repair(message: Dict[str, Any], tools: Optional[list] = None) -> RepairDecision:
    """Top-level decision for a full upstream assistant message.

    Handles native tool_calls and legacy function_call before falling back to
    content repair, so the server has a single entry point and one reason code.
    """
    message = message or {}
    if not tools:
        return RepairDecision(REASON_PASS_THROUGH_NO_TOOLS_DECLARED)

    # 1. Upstream already returned valid tool_calls — pass through unchanged.
    native = message.get("tool_calls")
    if native:
        return RepairDecision(REASON_VALID_NATIVE, list(native))

    # 2. Legacy OpenAI function_call field — convert, then validate like any
    #    recovered call so a missing required arg still abstains.
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        calls = [_call_from_legacy_function_call(function_call)]
        bad = _validate_calls(calls, tools)
        if bad:
            return RepairDecision(bad)
        return RepairDecision(REASON_REPAIRED_LEGACY_FUNCTION_CALL, calls)

    # 3. Content repair (the common small-model failure mode).
    return decide_repair(message.get("content") or "", tools)


def synthesize_tool_call_response(
    original_content: str,
    tool_calls: List[Dict[str, Any]],
    finish_reason: str = "tool_calls",
) -> Dict[str, Any]:
    """
    Build a response chunk (or full message) that looks like a normal
    OpenAI tool_calls response, while preserving the original content
    as reasoning if desired.
    """
    return {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": original_content if not tool_calls else None,  # or keep as reasoning
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }]
    }


# === Streaming support (the hard part for stock OpenCode) ===

class StreamingToolCallAccumulator:
    """
    Accumulates streaming content deltas for compat models.

    When it detects that the model has emitted a complete tool call as JSON
    inside the content (very common with Gemma 4 E4B), it can switch to
    emitting proper tool_calls instead.

    This is a pragmatic first version focused on making stock OpenCode work.
    It is not a perfect incremental argument streamer yet.
    """

    def __init__(self, model: str, known_tool_names: Optional[List[str]] = None):
        self.model = model
        self.known_tool_names = known_tool_names or []
        self.accumulated_content: str = ""
        self.switched_to_tool_call: bool = False
        self.tool_calls: Optional[List[Dict[str, Any]]] = None
        self.emitted_tool_call_chunks: int = 0

    def process_content_delta(self, delta_content: str) -> Optional[Dict[str, Any]]:
        """
        Feed a content delta. Returns a synthetic chunk to emit instead of
        the original content delta if we decide to switch to tool_calls mode.
        """
        if self.switched_to_tool_call:
            return None  # already switched, caller should stop sending content

        self.accumulated_content += delta_content or ""

        if not self.tool_calls and looks_like_tool_call_json(self.accumulated_content):
            parsed = parse_tool_call_from_content(
                self.accumulated_content,
                known_tool_names=self.known_tool_names or None,
            )
            if parsed:
                self.tool_calls = parsed
                self.switched_to_tool_call = True
                # Emit the first tool call chunk immediately
                return self._make_tool_call_chunk(0, is_last=False)

        return None

    def get_final_tool_call_chunk(self) -> Optional[Dict[str, Any]]:
        """Call at the end of the stream if we switched modes."""
        if self.switched_to_tool_call and self.tool_calls:
            return {
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": self.tool_calls},
                    "finish_reason": "tool_calls"
                }]
            }
        return None

    def _make_tool_call_chunk(self, index: int, is_last: bool = False) -> Dict[str, Any]:
        """Create a proper OpenAI streaming tool call delta chunk."""
        tc = self.tool_calls[index] if self.tool_calls else {}
        return {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [tc]  # In real incremental we would delta the arguments
                },
                "finish_reason": "tool_calls" if is_last else None
            }]
        }


def is_tool_call_delta(chunk: Dict[str, Any]) -> bool:
    """Heuristic to detect whether a streaming chunk already contains tool call info."""
    try:
        delta = chunk["choices"][0]["delta"]
        return bool(delta.get("tool_calls")) or bool(delta.get("function_call"))
    except Exception:
        return False


def rewrite_stream_chunk(chunk: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """
    Legacy simple hook. Prefer StreamingToolCallAccumulator for real use.
    """
    return chunk
