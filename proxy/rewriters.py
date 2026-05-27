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
        r'<execute_tool>.*?</execute_tool>',               # another common variant
        r'\b\w+\s*\{',                                     # toolName{...}  (ToolBridge-style JSON fallback)
        r'\b\w+\s*\(\s*\{',                                # toolName({...})
    ]
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def _parse_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse various XML-ish tool call formats that small models invent."""
    patterns = [
        # <tool_code>tool_name(args...)</tool_code>
        r'<tool_code>\s*(\w+)\s*\((.*?)\)\s*</tool_code>',
        # <execute_tool>tool_name(args...)</execute_tool>
        r'<execute_tool>\s*(\w+)\s*\((.*?)\)\s*</execute_tool>',
        # <tool><name>xxx</name><arguments>...</arguments></tool>
        r'<tool>\s*<name>\s*(\w+)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</tool>',
        # <run_terminal_cmd ...> or similar loose tags we saw in rigid tests
        r'<(run_terminal_cmd|write_file|git_commit)\b[^>]*>(.*?)</\1>',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            name = match.group(1)
            args_str = match.group(2).strip()
            try:
                if args_str.startswith('{'):
                    args = json.loads(args_str)
                else:
                    args = {}
                    for pair in re.split(r',\s*', args_str):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            args[k.strip()] = v.strip().strip('"\'')
            except Exception:
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


def _parse_json_tool_call_fallback(text: str, known_tool_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    ToolBridge-style JSON fallback for models that output things like:
      run_terminal_cmd{"command": "ls"}
      write_file({...})
    """
    if not text:
        return None
    names = known_tool_names or []
    for name in names:
        # Look for name{ or name( {
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
            # Light cleanup (single quotes, unquoted keys, trailing commas) - same spirit as ToolBridge
            cleaned = re.sub(r"'", '"', json_str)
            cleaned = re.sub(r'(\w+)\s*:', r'"\1":', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            args = json.loads(cleaned)
            return {
                "id": f"call_{name[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                }
            }
        except Exception:
            continue
    return None


def parse_tool_call_from_content(
    content: str,
    known_tool_names: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort extraction of tool call(s) from a content string.
    Tries multiple strategies because small models are very creative with formats.
    `known_tool_names` (from the original tools[] in the request) greatly helps
    disambiguation — taken from ToolBridge's approach.
    """
    if not content:
        return None

    text = content.strip()
    if not looks_like_tool_call_json(text):
        return None

    calls: List[Dict[str, Any]] = []
    known = known_tool_names or []

    # Strategy 1: XML-style formats (very common with small models)
    xml_call = _parse_xml_tool_call(text)
    if xml_call:
        calls.append(xml_call)
        return calls

    # Strategy 1b: ToolBridge-inspired JSON fallback for name{...} / name({...}) patterns
    if known:
        json_fb = _parse_json_tool_call_fallback(text, known)
        if json_fb:
            calls.append(json_fb)
            return calls

    # Strategy 2: Fenced JSON or raw JSON object/array
    fenced = re.search(r'```(?:json)?\s*(.+?)\s*```', text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text

    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "tool_calls" in data or "tool_call" in data:
                items = data.get("tool_calls") or data.get("tool_call") or []
            else:
                items = [data]
        else:
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool") or item.get("function", {}).get("name")
            args = item.get("arguments") or item.get("parameters") or item.get("args") or item.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                calls.append({
                    "id": f"call_{len(calls)}_{name[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                    }
                })
        if calls:
            return calls
    except Exception:
        pass

    # Strategy 3: Last resort - regex for {"name": "...", "arguments": ...}
    match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', text, re.DOTALL)
    if match:
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except Exception:
            args = {"raw": match.group(2)}
        calls.append({
            "id": f"call_{len(calls)}_{name[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        return calls

    # Strategy 4: Very last resort - bare "toolName{json}" even without known names list
    # (helps when the caller didn't forward tool names)
    bare = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(\{.*\})', text, re.DOTALL)
    if bare:
        name = bare.group(1)
        try:
            args = json.loads(bare.group(2))
            calls.append({
                "id": f"call_{name[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
            return calls
        except Exception:
            pass

    return None if not calls else calls


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
