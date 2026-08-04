#!/usr/bin/env python3
"""Apply strict submission gates to a primary budgeted-MCTS evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reject(reason: str) -> dict:
    return {
        "status": "reject",
        "reason": reason,
        "formal_submission_replacement_authorized": False,
    }


def submission_gate(report: dict) -> dict:
    games = int(report["wins"]) + int(report["losses"]) + int(report["draws"])
    if games != 400:
        return reject("requires_400_games")
    if int(report.get("exceptions", 0)):
        return reject("exceptions_present")
    if int(report.get("illegal_actions", 0)):
        return reject("illegal_actions_present")
    if float(report.get("p95_decision_seconds", float("inf"))) > 0.035:
        return reject("p95_latency_above_35ms")
    non_draws = int(report["wins"]) + int(report["losses"])
    win_rate = int(report["wins"]) / max(1, non_draws)
    if win_rate < 0.55:
        return reject("win_rate_below_55_percent")
    return {
        "status": "pass",
        "reason": "all_submission_gates_passed",
        "formal_submission_replacement_authorized": False,
        "win_rate": win_rate,
        "p95_decision_seconds": float(report["p95_decision_seconds"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    gate = submission_gate(report)
    document = {"evaluation": report, "submission_gate": gate}
    rendered = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
