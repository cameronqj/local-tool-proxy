"""Deterministic unit tests for the schema-aware repair decision layer.

These pin the exact reason code the proxy assigns to each repair/abstention
decision (see proxy/rewriters.py). They exercise `decide_repair` /
`decide_message_repair` directly — no server, no network — so the claim
boundaries are fast and unambiguous to read.

The central guarantee under test: the proxy emits tool_calls ONLY for a
complete, in-schema call, and otherwise abstains with an explicit reason. It
never fabricates a call the model did not clearly and completely express.
"""

import pytest

from proxy.rewriters import (
    decide_repair,
    decide_message_repair,
    REASON_VALID_NATIVE,
    REASON_REPAIRED_JSON,
    REASON_REPAIRED_MARKDOWN,
    REASON_REPAIRED_XML,
    REASON_REPAIRED_TOOLNAME,
    REASON_REPAIRED_LEGACY_FUNCTION_CALL,
    REASON_ABSTAIN_NO_TOOL_INTENT,
    REASON_ABSTAIN_MISSING_REQUIRED_ARGS,
    REASON_ABSTAIN_INVALID_ARGUMENTS,
    REASON_ABSTAIN_UNKNOWN_TOOL,
    REASON_ABSTAIN_AMBIGUOUS_MULTIPLE_TOOLS,
    REASON_ABSTAIN_MALFORMED_UNRECOVERABLE_JSON,
    REASON_PASS_THROUGH_NO_TOOLS_DECLARED,
    REPAIR_REASONS,
    ABSTAIN_REASONS,
)


def _tool(name, required=None, props=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


WRITE_FILE = _tool("write_file", required=["path"], props={"path": {"type": "string"}})
RUN_COMMAND = _tool("run_command", required=["command"], props={"command": {"type": "string"}})
NO_REQ_TOOL = _tool("note", required=[], props={"text": {"type": "string"}})


# --- Repair decisions (the proxy emits tool_calls) -------------------------

def test_native_tool_calls_pass_through():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": '{"path": "a.py"}'}}
        ],
    }
    d = decide_message_repair(msg, [WRITE_FILE])
    assert d.reason == REASON_VALID_NATIVE
    assert d.repaired
    assert d.tool_calls[0]["function"]["name"] == "write_file"


def test_repaired_json_content():
    d = decide_repair('{"name": "write_file", "arguments": {"path": "a.py"}}', [WRITE_FILE])
    assert d.reason == REASON_REPAIRED_JSON
    assert len(d.tool_calls) == 1
    assert d.tool_calls[0]["function"]["name"] == "write_file"


def test_repaired_markdown_json():
    text = '```json\n{"name": "write_file", "arguments": {"path": "a.py"}}\n```'
    d = decide_repair(text, [WRITE_FILE])
    assert d.reason == REASON_REPAIRED_MARKDOWN
    assert d.repaired


def test_repaired_xml_tool_block():
    d = decide_repair('<tool_code>run_command("ls -la")</tool_code>', [RUN_COMMAND])
    assert d.reason == REASON_REPAIRED_XML
    assert d.tool_calls[0]["function"]["name"] == "run_command"


def test_repaired_toolname_snippet():
    d = decide_repair('run_command{"command": "pytest -q"}', [RUN_COMMAND])
    assert d.reason == REASON_REPAIRED_TOOLNAME
    assert d.tool_calls[0]["function"]["name"] == "run_command"


def test_repaired_legacy_function_call():
    msg = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "write_file", "arguments": '{"path": "a.py"}'},
    }
    d = decide_message_repair(msg, [WRITE_FILE])
    assert d.reason == REASON_REPAIRED_LEGACY_FUNCTION_CALL
    assert d.tool_calls[0]["function"]["name"] == "write_file"


# --- Abstentions (the proxy refuses to fabricate a call) -------------------

def test_abstain_no_tool_intent_on_prose():
    prose = "I will write_file after I inspect the project, then decide what to change."
    d = decide_repair(prose, [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_NO_TOOL_INTENT
    assert d.tool_calls == []


def test_abstain_missing_required_args():
    d = decide_repair('{"name": "write_file", "arguments": {}}', [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_MISSING_REQUIRED_ARGS
    assert d.tool_calls == []


def test_abstain_invalid_arguments_empty_value():
    d = decide_repair('{"name": "write_file", "arguments": {"path": ""}}', [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_INVALID_ARGUMENTS
    assert d.tool_calls == []


def test_abstain_unknown_tool():
    d = decide_repair('{"name": "delete_everything", "arguments": {"x": 1}}', [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_UNKNOWN_TOOL
    assert d.tool_calls == []


def test_abstain_ambiguous_multiple_tools():
    # Tool-ish but unparseable, and two declared tools are named without one
    # clear invocation.
    text = '{"name": "write_file" ... or maybe run_command, broken json'
    d = decide_repair(text, [WRITE_FILE, RUN_COMMAND])
    assert d.reason == REASON_ABSTAIN_AMBIGUOUS_MULTIPLE_TOOLS
    assert d.tool_calls == []


def test_abstain_malformed_unrecoverable_json():
    text = '{"name": "write_file" this is broken and unrecoverable'
    d = decide_repair(text, [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_MALFORMED_UNRECOVERABLE_JSON
    assert d.tool_calls == []


def test_abstain_on_documentation_example_for_undeclared_tool():
    # A model that echoes example/documentation JSON for a tool that was never
    # declared must not have it turned into a real call.
    text = '{"name": "example_tool", "arguments": {"foo": "bar"}}'
    d = decide_repair(text, [WRITE_FILE])
    assert d.reason == REASON_ABSTAIN_UNKNOWN_TOOL
    assert d.tool_calls == []


# --- Pass-through ----------------------------------------------------------

def test_pass_through_when_no_tools_declared():
    # Even content that parses as a tool call must not become one when the
    # request declared no tools to validate against.
    d = decide_repair('{"name": "write_file", "arguments": {"path": "a.py"}}', [])
    assert d.reason == REASON_PASS_THROUGH_NO_TOOLS_DECLARED
    assert d.tool_calls == []

    msg = {"role": "assistant", "content": "hello"}
    assert decide_message_repair(msg, []).reason == REASON_PASS_THROUGH_NO_TOOLS_DECLARED


# --- Invariants ------------------------------------------------------------

def test_tool_with_no_required_args_repairs_cleanly():
    d = decide_repair('{"name": "note", "arguments": {}}', [NO_REQ_TOOL])
    assert d.reason == REASON_REPAIRED_JSON
    assert d.repaired


@pytest.mark.parametrize("reason", sorted(ABSTAIN_REASONS))
def test_abstain_reasons_never_emit_calls(reason):
    # Sanity: the abstain code set and repair code set are disjoint.
    assert reason not in REPAIR_REASONS
