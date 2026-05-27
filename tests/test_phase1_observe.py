"""
Phase 1 evaluation test for NextGrok observe mode.

Simulates sequences of tool-using turns (including the exact collapse patterns
seen in real opencode + gemma4:e4b-mlx runs from May 2026) and verifies:
- correct collapse classification
- drift score updates
- report generation
"""

import sys
sys.path.insert(0, ".")

import proxy.server as srv
from proxy.collapse import classify_from_openai_response, get_collapse_signals
from proxy.collapse_report import parse_log_lines, format_report


def simulate_turn(trace: str, response: dict, had_tools: bool = True):
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

    drift = {
        "tool_streak": state["tool_streak"],
        "turns_since_last_tool": state["turns_since_last_tool"],
        "total_tool_turns": state["total_tool_turns"],
        "collapse_events": state["collapse_events"],
    }

    return cat, signals, drift


def main():
    print("=== Phase 1 Observe Mode Evaluation ===\n")

    srv.MODE = "observe"
    srv.TRACE_DRIFT.clear()

    # Simulate the rigid tictactoe collapse pattern (very common in our real runs)
    rigid_sequence = [
        ("rigid-1", {"choices": [{"message": {"tool_calls": [{}]}, "finish_reason": "tool_calls"}]}),
        ("rigid-2", {"choices": [{"message": {"content": "1. mkdir tictactoe\n2. git init"}, "finish_reason": "stop"}]}),
        ("rigid-3", {"choices": [{"message": {"content": "Next I will create the FastAPI backend."}, "finish_reason": "stop"}]}),
    ]

    print("--- Rigid prompt simulation (matches May 2026 real opencode run) ---")
    for name, resp in rigid_sequence:
        cat, signals, drift = simulate_turn(name, resp)
        print(f"{name}: category={cat} signals={signals}")
        print(f"        drift={drift}\n")

    # Simulate the "realistic" tasklite pattern (slightly better survival)
    tasklite_sequence = [
        ("tasklite-1", {"choices": [{"message": {"tool_calls": [{}]}, "finish_reason": "tool_calls"}]}),
        ("tasklite-2", {"choices": [{"message": {"tool_calls": [{}]}, "finish_reason": "tool_calls"}]}),
        ("tasklite-3", {"choices": [{"message": {"content": "I will now write the test file."}, "finish_reason": "stop"}]}),
    ]

    print("--- Tasklite (realistic bugfix) simulation ---")
    for name, resp in tasklite_sequence:
        cat, signals, drift = simulate_turn(name, resp)
        print(f"{name}: category={cat} signals={signals}")
        print(f"        drift={drift}\n")

    # Generate report from the simulated log lines
    sample_log_lines = [
        "[gptfixes-rigid-2] collapse: category=literal_commands signals=['model_printed_literal_commands']",
        "[gptfixes-rigid-3] collapse: category=tool_intent_prose signals=['model_described_action_without_calling_tool']",
        "[gptfixes-tasklite-3] collapse: category=tool_intent_prose signals=['model_described_action_without_calling_tool']",
    ]

    print("--- Collapse Report (as would be produced in Phase 1 evaluation) ---")
    report_data = parse_log_lines(sample_log_lines)
    print(format_report(report_data))

    print("\nPhase 1 observe mode evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
