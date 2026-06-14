"""Basic Phase 2 rewriter tests using real failure examples we've seen."""

import json
from pathlib import Path
from proxy.rewriters import (
    parse_json_content,
    parse_tool_call_from_content,
    parse_xml_content,
    repair_jsonish,
    extract_known_tool_names,
    looks_like_tool_call_json,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_outputs"

def test_extract_known_tool_names():
    payload = {
        "tools": [
            {"type": "function", "function": {"name": "run_terminal_cmd"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
    }
    assert extract_known_tool_names(payload) == ["run_terminal_cmd", "write_file"]


def test_repair_jsonish_and_parse_json_content():
    """repair_jsonish is a helper; the real contract is exercised via parse_json_content."""
    bad = "run_terminal_cmd{'command': 'ls -la',}"
    calls = parse_json_content(bad, ["run_terminal_cmd"])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_terminal_cmd"
    assert calls[0]["function"]["arguments"]  # should be valid JSON string after repair + parse


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


def test_realistic_fixture_outputs_parse():
    cases = [
        ("json_in_content.json", ["write_file"], "write_file"),
        ("tool_name_jsonish.txt", ["run_terminal_cmd"], "run_terminal_cmd"),
        ("xml_tool_code.txt", ["run_command"], "run_command"),
    ]

    for filename, known_tools, expected_name in cases:
        text = (FIXTURE_DIR / filename).read_text()
        calls = parse_tool_call_from_content(text, known_tools)
        assert calls, f"Expected fixture to parse: {filename}"
        assert calls[0]["function"]["name"] == expected_name


def test_prose_fixture_does_not_parse_as_tool_call():
    text = (FIXTURE_DIR / "prose_false_positive.txt").read_text()
    assert parse_tool_call_from_content(text, ["write_file"]) is None


def test_valid_json_with_colon_and_apostrophe_values_is_not_corrupted():
    """Regression: repair_jsonish must not run on already-valid JSON. A URL
    (embedded ':') or an apostrophe in a value previously broke parsing, so the
    recoverable call was silently dropped."""
    url_call = '{"name": "fetch", "arguments": {"url": "http://example.com:8080/a"}}'
    calls = parse_tool_call_from_content(url_call, ["fetch"])
    assert calls and calls[0]["function"]["name"] == "fetch"
    assert json.loads(calls[0]["function"]["arguments"])["url"] == "http://example.com:8080/a"

    apos_call = '{"name": "write_file", "arguments": {"path": "a.py", "content": "it' + chr(39) + 's ok"}}'
    calls = parse_tool_call_from_content(apos_call, ["write_file"])
    assert calls and json.loads(calls[0]["function"]["arguments"])["content"] == "it's ok"
