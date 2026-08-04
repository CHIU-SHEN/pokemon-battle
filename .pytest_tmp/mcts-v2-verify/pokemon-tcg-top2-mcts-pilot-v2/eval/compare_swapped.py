#!/usr/bin/env python3
"""Combine two swapped-side match summaries for one candidate-versus-baseline Arena."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stats import wilson_interval


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_first", type=Path, help="summary where candidate is agent0")
    parser.add_argument("candidate_second", type=Path, help="summary where candidate is agent1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first, second = load(args.candidate_first), load(args.candidate_second)
    wins = int(first["agent0_wins"]) + int(second["agent1_wins"])
    losses = int(first["agent1_wins"]) + int(second["agent0_wins"])
    draws = int(first["draws"]) + int(second["draws"])
    games = wins + losses + draws
    low, high = wilson_interval(wins, games)
    sources = {}
    for summary, side in ((first, 0), (second, 1)):
        for name, count in summary.get("action_sources", [{}, {}])[side].items():
            sources[name] = sources.get(name, 0) + int(count)
    decisions = sum(sources.values())
    gru_calls = sources.get("gru", 0)
    report = {
        "schema_version": "swapped_arena_summary_v1",
        "candidate": first.get("agent0"),
        "baseline": first.get("agent1"),
        "seed": first.get("seed"),
        "engine_seed_controlled": bool(first.get("engine_seed_controlled", False)),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "wilson_95": [low, high],
        "exceptions": int(first.get("exceptions", 0)) + int(second.get("exceptions", 0)),
        "illegal_actions": sum(first.get("illegal_actions", [0, 0])) + sum(second.get("illegal_actions", [0, 0])),
        "p95_decision_seconds_upper": max(
            float(first.get("p95_agent_time_sec_per_decision", 0.0)),
            float(second.get("p95_agent_time_sec_per_decision", 0.0)),
        ),
        "candidate_action_sources": sources,
        "candidate_model_action_rate": gru_calls / decisions if decisions else None,
        "inputs": [str(args.candidate_first), str(args.candidate_second)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
