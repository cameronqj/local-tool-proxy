"""Observe-mode collapse classification tests.

Exercises the collapse classifier and drift accounting against the exact
sequences seen in real opencode + gemma4:e4b-mlx runs (May 2026), plus the
log-scanning report generator. These are real `assert`-based pytest tests (the
file was previously a `print`/`main()` script that pytest collected as zero
tests).
"""

import pytest

import proxy.server as srv
from proxy.collapse import classify_from_openai_response, get_collapse_signals
from proxy.collapse_report import parse_log_lines, format_report


def _simulate_turn(trace: str, response: dict, had_tools: bool = True):
    cat = classify_from_openai_response(response, had_tools_in_request=had_tools)
    signals = get_collapse_signals(cat, had_tools)

    state = srv.TRACE_DRIFT[trace]
    if cat == "tool_calls":
        state["tool_streak"] += 1
        state["turns_since_last_tool"] = 0
        state["total_tool_turns"] += 1
    else:
        state["tool_streak"] = 0
        state["turns_since_last_tool"] += 1
        state["collapse_events"] += 1
    return cat, signals, dict(state)


@pytest.fixture(autouse=True)
def _clean_drift_state():
    srv.TRACE_DRIFT.clear()
    yield
    srv.TRACE_DRIFT.clear()


def _tool_turn():
    return {"choices": [{"message": {"tool_calls": [{}]}, "finish_reason": "tool_calls"}]}


def _content_turn(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def test_rigid_prompt_collapse_sequence_is_classified():
    """Matches the May 2026 real opencode run: one tool turn, then the model
    drifts into literal commands and then tool-intent prose."""
    cat1, _, drift1 = _simulate_turn("rigid-1", _tool_turn())
    assert cat1 == "tool_calls"
    assert drift1["tool_streak"] == 1

    cat2, sig2, drift2 = _simulate_turn("rigid-2", _content_turn("1. mkdir tictactoe\n2. git init"))
    assert cat2 == "literal_commands"
    assert sig2 == ["model_printed_literal_commands"]
    assert drift2["tool_streak"] == 0 and drift2["collapse_events"] == 1

    cat3, sig3, _ = _simulate_turn("rigid-3", _content_turn("Next I will create the FastAPI backend."))
    assert cat3 == "tool_intent_prose"
    assert sig3 == ["model_described_action_without_calling_tool"]


def test_tasklite_sequence_survives_longer_then_drifts():
    cats = [
        _simulate_turn("tasklite-1", _tool_turn())[0],
        _simulate_turn("tasklite-2", _tool_turn())[0],
        _simulate_turn("tasklite-3", _content_turn("I will now write the test file."))[0],
    ]
    assert cats == ["tool_calls", "tool_calls", "tool_intent_prose"]
    assert srv.TRACE_DRIFT["tasklite-2"]["tool_streak"] == 1


def test_collapse_report_aggregates_log_lines():
    log_lines = [
        "[ltp-rigid-2] collapse: category=literal_commands signals=['model_printed_literal_commands']",
        "[ltp-rigid-3] collapse: category=tool_intent_prose signals=['model_described_action_without_calling_tool']",
        "[ltp-tasklite-3] collapse: category=tool_intent_prose signals=['model_described_action_without_calling_tool']",
    ]
    data = parse_log_lines(log_lines)
    assert data["total_traces"] == 3
    assert data["category_counts"]["tool_intent_prose"] == 2
    assert data["category_counts"]["literal_commands"] == 1

    report = format_report(data)
    assert "Collapse Report" in report
    assert "tool_intent_prose" in report
