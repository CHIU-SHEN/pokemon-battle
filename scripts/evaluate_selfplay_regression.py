#!/usr/bin/env python3
"""Evaluate history, weak-baseline, cross-Top2, and latency promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_match import first_min_agent, play_game, random_agent, summarize, with_deck  # noqa: E402
from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.arena.ppo_agent import PPOArenaAgent  # noqa: E402
from src.rl.selfplay_regression import regression_decision  # noqa: E402
from src.rl.selfplay_state import SelfPlayState  # noqa: E402


def arena(
    candidate,
    opponent,
    *,
    candidate_deck: list[int],
    opponent_deck: list[int],
    games: int,
    max_steps: int,
    seed: int,
    opponent_name: str,
) -> dict:
    records = []
    for index in range(games):
        side = index % 2
        left = with_deck(candidate if side == 0 else opponent, candidate_deck if side == 0 else opponent_deck)
        right = with_deck(opponent if side == 0 else candidate, opponent_deck if side == 0 else candidate_deck)
        record = play_game(left, right, max_steps)
        if side == 1 and record["result"] in (0, 1):
            record["result"] = 1 - record["result"]
        records.append(record)
    return summarize(records, "candidate", opponent_name, seed)


def rate(report: dict) -> float:
    non_draw = int(report["agent0_wins"]) + int(report["agent1_wins"])
    return int(report["agent0_wins"]) / max(non_draw, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--selfplay-root", type=Path, required=True)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    branches = {item["role"]: item for item in config["branches"]}
    branch = branches[args.branch]
    other = branches["reserve" if args.branch == "primary" else "primary"]
    project_root = args.project_root.resolve()
    state = SelfPlayState.load(
        args.selfplay_root.resolve() / args.branch,
        expected_branch=args.branch,
        expected_deck_id=branch["deck_id"],
    )
    candidate = PPOArenaAgent(
        branch["candidate_id"],
        branch["deck_id"],
        args.candidate_checkpoint.resolve(),
        project_root=project_root,
        device=args.device,
    )
    if state.best.get("checkpoint_kind") == "ppo":
        best = PPOArenaAgent(
            branch["candidate_id"],
            branch["deck_id"],
            Path(state.best["path"]),
            project_root=project_root,
            device=args.device,
        )
    else:
        best = AdapterArenaAgent(branch["candidate_id"], project_root=project_root, device=args.device)
    cross = AdapterArenaAgent(other["candidate_id"], project_root=project_root, device=args.device)

    weak_reports = [
        arena(
            candidate,
            opponent,
            candidate_deck=list(candidate.deck),
            opponent_deck=list(candidate.deck),
            games=args.games,
            max_steps=args.max_steps,
            seed=args.seed + index,
            opponent_name=name,
        )
        for index, (name, opponent) in enumerate((("random", random_agent), ("first-min", first_min_agent)))
    ]
    history_reports = []
    for index, item in enumerate(state.history[-3:]):
        history_agent = PPOArenaAgent(
            branch["candidate_id"],
            branch["deck_id"],
            Path(item["path"]),
            project_root=project_root,
            device=args.device,
        )
        history_reports.append(
            arena(
                candidate,
                history_agent,
                candidate_deck=list(candidate.deck),
                opponent_deck=list(history_agent.deck),
                games=args.games,
                max_steps=args.max_steps,
                seed=args.seed + 10 + index,
                opponent_name=f"history-{index}",
            )
        )
    candidate_cross = arena(
        candidate,
        cross,
        candidate_deck=list(candidate.deck),
        opponent_deck=list(cross.deck),
        games=args.games,
        max_steps=args.max_steps,
        seed=args.seed + 20,
        opponent_name="cross-top2",
    )
    best_cross = arena(
        best,
        cross,
        candidate_deck=list(best.deck),
        opponent_deck=list(cross.deck),
        games=args.games,
        max_steps=args.max_steps,
        seed=args.seed + 21,
        opponent_name="cross-top2",
    )
    illegal_actions = sum(
        sum(report["illegal_actions"])
        for report in weak_reports + history_reports + [candidate_cross, best_cross]
    )
    metrics = {
        "illegal_actions": illegal_actions,
        "recent_history_win_rate": (
            sum(rate(report) for report in history_reports) / len(history_reports)
            if history_reports
            else None
        ),
        "weak_baseline_win_rate": sum(rate(report) for report in weak_reports) / len(weak_reports),
        "candidate_cross_win_rate": rate(candidate_cross),
        "best_cross_win_rate": rate(best_cross),
        "candidate_p95_seconds": float(candidate_cross["p95_agent_time_sec_per_decision"]),
        "best_p95_seconds": float(best_cross["p95_agent_time_sec_per_decision"]),
    }
    decision = regression_decision(metrics)
    report = {
        "schema_version": "top2_selfplay_regression_v1",
        "role": args.branch,
        **decision,
        "weak_reports": weak_reports,
        "history_reports": history_reports,
        "candidate_cross": candidate_cross,
        "best_cross": best_cross,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
