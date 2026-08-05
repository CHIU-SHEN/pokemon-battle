#!/usr/bin/env python3
"""Audit a complete primary MCTS collection and print cumulative JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.mcts_collection import audit_collection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_collection(args.root, {"branch": "primary", "deck_id": args.deck_id}, require_all_splits=True)
    if report["games"] != args.expected_games:
        raise ValueError(f"expected {args.expected_games} games, found {report['games']}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
