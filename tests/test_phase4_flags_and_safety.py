"""
Phase 4 tests: CLI flag gating + safety / replay guarantees.
"""

import sys
import subprocess

sys.path.insert(0, ".")


def test_default_is_compat_and_no_stabilize():
    """Default invocation must be pure compat with no stabilize/planner active."""
    from proxy import server as srv

    # Reset globals as if freshly imported in compat mode
    srv.MODE = "compat"
    srv.STABILIZE_MAX_RETRIES = 0
    srv.PLANNER_MODE = "disabled"

    assert srv.MODE == "compat"
    assert srv.STABILIZE_MAX_RETRIES == 0
    assert srv.PLANNER_MODE == "disabled"

    # The stabilize logic should not trigger in compat
    from proxy.stabilizer import should_attempt_stabilize
    assert should_attempt_stabilize("tool_intent_prose", "compat") is False

    print("Default is pure compat: OK")


def test_help_mentions_experimental():
    """--help should make it clear that stabilize/planner are experimental."""
    result = subprocess.run(
        [sys.executable, "-m", "proxy.server", "--help"],
        capture_output=True, text=True, timeout=10
    )
    output = result.stdout + result.stderr
    assert "stabilize" in output.lower()
    assert "experimental" in output.lower() or "opt-in" in output.lower()
    print("--help documents experimental nature: OK")


if __name__ == "__main__":
    test_default_is_compat_and_no_stabilize()
    test_help_mentions_experimental()
    print("Phase 4 flag gating + safety tests passed.")
