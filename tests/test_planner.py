"""
Phase 3 smoke test for Soft Planner integration.
"""

import sys
sys.path.insert(0, ".")

from proxy.planner import extract_milestones, build_planner_hint, should_use_planner_for_recovery


def test_planner_basic():
    msg = "Fix the bug in buggy.py and write tests that demonstrate the fix."
    tools = ["read_file", "write_file", "run_command"]

    agenda = extract_milestones(msg, tools)
    assert len(agenda) >= 1
    print("Agenda extracted:", agenda)

    hint = build_planner_hint(agenda, {"files_edited": 1, "tests_passing": False})
    print("Planner hint:", hint)
    assert "milestones" in hint.lower() or "progress" in hint.lower()

    assert should_use_planner_for_recovery("soft", True) is True
    assert should_use_planner_for_recovery("soft", False) is False
    assert should_use_planner_for_recovery("disabled", True) is False

    print("Phase 3 planner basic test passed.")


if __name__ == "__main__":
    test_planner_basic()
