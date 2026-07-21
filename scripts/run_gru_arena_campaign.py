#!/usr/bin/env python3
"""Run a resumable, time-budgeted local GRU checkpoint Arena campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "run_match.py"
BASELINE = ROOT / "final_submissions" / "sl0_shared_stage1" / "main.py"
CANDIDATES = {
    "seed1_best": ROOT / "final_submissions" / "sl1_gru_seed1_best" / "main.py",
    "seed1_last": ROOT / "final_submissions" / "sl1_gru_seed1_last" / "main.py",
    "seed2_epoch5": ROOT / "final_submissions" / "sl1_gru_seed2_epoch5" / "main.py",
}


def wilson(wins: int, games: int, z: float = 1.96) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    import math
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_leg(candidate: Path, candidate_first: bool, games: int, label: str, root: Path) -> dict:
    out = root / label
    summary_path = out / "summary.json"
    if summary_path.exists():
        return load(summary_path)
    agent0, agent1 = (candidate, BASELINE) if candidate_first else (BASELINE, candidate)
    command = [
        sys.executable, str(EVAL), "--agent0", str(agent0), "--agent1", str(agent1),
        "--games", str(games), "--seed", "20260721", "--out-dir", str(out),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return load(summary_path)


def candidate_result(name: str, summaries: list[tuple[dict, int]]) -> dict:
    wins = losses = draws = exceptions = illegal = decisions = model_actions = 0
    p95 = 0.0
    for summary, side in summaries:
        wins += int(summary["agent0_wins"] if side == 0 else summary["agent1_wins"])
        losses += int(summary["agent1_wins"] if side == 0 else summary["agent0_wins"])
        draws += int(summary["draws"])
        exceptions += int(summary.get("exceptions", 0))
        illegal += sum(int(x) for x in summary.get("illegal_actions", [0, 0]))
        p95 = max(p95, float(summary.get("p95_agent_time_sec_per_decision", 0.0)))
        sources = summary.get("action_sources", [{}, {}])[side]
        decisions += sum(int(value) for value in sources.values())
        model_actions += int(sources.get("gru", 0))
    games = wins + losses + draws
    return {
        "candidate": name, "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / games if games else 0.0, "wilson_95": wilson(wins, games),
        "exceptions": exceptions, "illegal_actions": illegal,
        "p95_decision_seconds_upper": p95,
        "model_action_rate": model_actions / decisions if decisions else None,
    }


def write_report(path: Path, started: float, deadline: float, phase: str,
                 results: dict[str, list[tuple[dict, int]]], winner: str | None) -> None:
    aggregates = {name: candidate_result(name, summaries) for name, summaries in results.items()}
    report = {
        "schema_version": "gru_arena_campaign_v1",
        "phase": phase,
        "engine_seed_controlled": False,
        "started_at_epoch": started,
        "deadline_epoch": deadline,
        "elapsed_seconds": time.time() - started,
        "winner_after_screening": winner,
        "results": aggregates,
        "ranking": sorted(aggregates, key=lambda key: aggregates[key]["win_rate"], reverse=True),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=2.75)
    parser.add_argument("--screen-games-per-side", type=int, default=400)
    parser.add_argument("--final-games-per-side", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/gru_arena_campaign_20260721")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "campaign_report.json"
    started = time.time()
    deadline = started + args.hours * 3600
    results: dict[str, list[tuple[dict, int]]] = {name: [] for name in CANDIDATES}

    for name, candidate in CANDIDATES.items():
        for side in (0, 1):
            summary = run_leg(
                candidate, side == 0, args.screen_games_per_side,
                f"screen_{name}_{'first' if side == 0 else 'second'}", args.out_dir,
            )
            results[name].append((summary, side))
            write_report(report_path, started, deadline, "screening", results, None)

    winner = max(results, key=lambda name: candidate_result(name, results[name])["win_rate"])
    write_report(report_path, started, deadline, "finalist", results, winner)
    round_index = 0
    # Reserve twice the most recent leg duration conservatively by stopping 10 minutes early.
    while time.time() < deadline - 600:
        for side in (0, 1):
            before = time.time()
            summary = run_leg(
                CANDIDATES[winner], side == 0, args.final_games_per_side,
                f"final_{winner}_{round_index:03d}_{'first' if side == 0 else 'second'}", args.out_dir,
            )
            results[winner].append((summary, side))
            write_report(report_path, started, deadline, "finalist", results, winner)
            if time.time() >= deadline - max(600, 2 * (time.time() - before)):
                break
        round_index += 1
    write_report(report_path, started, deadline, "complete", results, winner)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
