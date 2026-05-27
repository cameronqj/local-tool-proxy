from __future__ import annotations

"""
Rewriters / adapters for small local models.

Goal: Turn the messy but often-correct output of Gemma 4 E*, gpt-oss, etc.
into the clean streaming tool_calls format that normal OpenCode-like harnesses expect.

This module is intentionally small and focused. It is the heart of what makes
the in-line proxy valuable for the 24 GB fanless M4 Air use case.

Current status: skeleton + first "json in content" detector/synthesizer.
More parsers (XML-ish, single-tool-text, legacy function_call, reasoning_content stripping)
will be added here, drawing from:
  - SmallHarness fallback logic (looks_like_start_of_tool_call, try_parse_inline_tool_call, etc.)
  - OpenCode PR #16531 toolParser (json / raw-function-call / single-tool-text + SSE rewriting)
  - Real failure traces we collected earlier in this repo
"""

import json
import re

from typing import Any, Dict, List, Optional, Tuple


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
            # becomes {"command": 'mkdir foo'}  (best effort)
            args = {"raw": args_str}
            # Heuristic: if it looks like a single quoted string, extract the content as "command"
            if (args_str.startswith('"') and args_str.endswith('"')) or (args_str.startswith("'") and args_str.endswith("'")):
                args = {"command": args_str[1:-1]}

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
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool") or (item.get("function") or {}).get("name")
            if not name:
                continue
            args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

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
    Best-effort extraction using a clear strategy list (Phase 2 architecture per gptfixes.prompt).

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
