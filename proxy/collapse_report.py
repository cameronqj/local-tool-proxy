#!/usr/bin/env python3
"""
proxy/collapse_report.py

Simple report generator that scans proxy log output (or future trace files)
and summarizes collapse patterns.

Usage examples:
    python -m proxy.collapse_report --log-file /path/to/proxy.log
    python -m proxy.collapse_report --from-stdin < proxy-output.txt
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


COLLAPSE_LINE_RE = re.compile(
    r"\[(?P<trace>(?:ltp|gptfixes)-[a-f0-9]+)\] collapse: category=(?P<cat>\w+) signals=\[(?P<signals>[^\]]*)\]"
)
# Also support the slightly different format used in some log examples
COLLAPSE_LINE_RE2 = re.compile(
    r"\[(?P<trace>(?:ltp|gptfixes)-[^\]]+)\] collapse: category=(?P<cat>\w+)"
)


def parse_log_lines(lines: List[str]) -> Dict[str, any]:
    """Parse collapse classification lines from proxy logs."""
    reports: Dict[str, Dict] = defaultdict(lambda: {"categories": Counter(), "signals": Counter(), "traces": 0})

    for line in lines:
        m = COLLAPSE_LINE_RE.search(line)
        if not m:
            m = COLLAPSE_LINE_RE2.search(line)
        if not m:
            continue

        trace = m.group("trace")
        cat = m.group("cat")
        signals_raw = m.group("signals", "") if "signals" in m.groupdict() else ""

        reports[trace]["categories"][cat] += 1
        reports[trace]["traces"] = 1

        if signals_raw.strip():
            for sig in signals_raw.split(","):
                sig = sig.strip().strip("'\"[] ")
                if sig:
                    reports[trace]["signals"][sig] += 1

    # Aggregate across all traces
    total = {
        "total_traces": len(reports),
        "category_counts": Counter(),
        "signal_counts": Counter(),
        "per_trace": dict(reports),
    }

    for data in reports.values():
        total["category_counts"].update(data["categories"])
        total["signal_counts"].update(data["signals"])

    return total


def format_report(data: Dict) -> str:
    lines = []
    lines.append("=== local-tool-proxy Collapse Report ===")
    lines.append(f"Traces analyzed: {data['total_traces']}")
    lines.append("")

    lines.append("Category distribution:")
    for cat, count in data["category_counts"].most_common():
        lines.append(f"  {cat:25s} {count}")
    lines.append("")

    lines.append("Top collapse signals:")
    for sig, count in data["signal_counts"].most_common(15):
        lines.append(f"  {sig:40s} {count}")
    lines.append("")

    # Show a few example traces with interesting signals
    interesting = [
        (t, d) for t, d in data["per_trace"].items()
        if any("no_tool" in s or "literal" in s or "prose" in s for s in d["signals"])
    ][:5]

    if interesting:
        lines.append("Example traces with collapse signals:")
        for trace_id, d in interesting:
            cats = ", ".join(f"{c}:{n}" for c, n in d["categories"].items())
            lines.append(f"  {trace_id}: {cats}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="local-tool-proxy collapse report generator")
    parser.add_argument("--log-file", type=Path, help="Path to proxy log file")
    parser.add_argument("--from-stdin", action="store_true", help="Read log lines from stdin")
    args = parser.parse_args()

    lines: List[str] = []

    if args.log_file:
        lines = args.log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    elif args.from_stdin:
        lines = sys.stdin.read().splitlines()
    else:
        parser.print_help()
        return 1

    data = parse_log_lines(lines)
    print(format_report(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
