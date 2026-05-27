"""Basic Phase 2 rewriter tests using real failure examples we've seen."""

import json
from proxy.rewriters import (
    parse_json_content,
    parse_xml_content,
    repair_jsonish,
    extract_known_tool_names,
    looks_like_tool_call_json,
)

def test_extract_known_tool_names():
    payload = {
        "tools": [
            {"type": "function", "function": {"name": "run_terminal_cmd"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
    }
    assert extract_known_tool_names(payload) == ["run_terminal_cmd", "write_file"]


def test_repair_jsonish():
    bad = "run_terminal_cmd{'command': 'ls -la',}"
    cleaned = repair_jsonish(bad)
    # The function should at minimum make it parseable JSON
    data = json.loads(cleaned)
    assert data.get("command") == "ls -la"


def test_parse_json_content_with_known_names():
    text = 'run_terminal_cmd{"command": "ls -la"}'
    calls = parse_json_content(text, ["run_terminal_cmd"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_terminal_cmd"


def test_parse_xml_tool_code():
    text = '<tool_code>run_terminal_cmd("ls -la")</tool_code>'
    calls = parse_xml_content(text, ["run_terminal_cmd"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_terminal_cmd"

def test_parse_xml_tool_code_with_run_command_style():
    # Exact format observed in recent Test A run
    text = '<tool_code>run_command("mkdir multi_tool_test")</tool_code>'
    calls = parse_xml_content(text, ["run_command"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_command"
    assert "mkdir multi_tool_test" in calls[0]["function"]["arguments"]


def test_looks_like_detects_variants():
    assert looks_like_tool_call_json('**Tool Call:** run_terminal_cmd{"command": "ls"}')
    assert looks_like_tool_call_json('<execute_tool name="write_file">...</execute_tool>')
