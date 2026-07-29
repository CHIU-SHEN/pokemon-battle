#!/usr/bin/env python3
"""Collect branch-isolated Top2 rollouts with swapped seats."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_match import first_min_agent, play_game, random_agent, with_deck  # noqa: E402
from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.rl.top2_rollout import Top2RolloutAgent, finalize_trajectory, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve", "all"), default="all")
    parser.add_argument("--opponents", default="cross-top2,first-min,random")
    parser.add_argument("--games-per-opponent", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/adapter_top2_rl_rollouts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.games_per_opponent <= 0:
        raise ValueError("games-per-opponent must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    project_root = args.project_root.resolve()
    branches = {branch["role"]: branch for branch in config["branches"]}
    selected = list(branches) if args.branch == "all" else [args.branch]
    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    unknown = set(opponents) - {"cross-top2", "first-min", "random"}
    if unknown:
        raise ValueError(f"unknown opponents: {sorted(unknown)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_id
    random.seed(args.seed)
    summaries = []
    for role in selected:
        branch = branches[role]
        other = branches["reserve" if role == "primary" else "primary"]
        learner = Top2RolloutAgent(
            branch["candidate_id"],
            branch["deck_id"],
            project_root=project_root,
            device=args.device,
            seed=args.seed + (0 if role == "primary" else 100000),
            temperature=args.temperature,
        )
        branch_games = []
        for opponent_name in opponents:
            if opponent_name == "cross-top2":
                opponent = AdapterArenaAgent(other["candidate_id"], project_root=project_root, device=args.device)
                opponent_deck = list(opponent.deck)
            elif opponent_name == "first-min":
                opponent = first_min_agent
                opponent_deck = list(learner.deck)
            else:
                opponent = random_agent
                opponent_deck = list(learner.deck)
            for game_index in range(args.games_per_opponent):
                side = game_index % 2
                learner.reset_trajectory()
                learner_wrapped = with_deck(learner, learner.deck)
                opponent_wrapped = with_deck(opponent, opponent_deck)
                record = play_game(
                    learner_wrapped if side == 0 else opponent_wrapped,
                    opponent_wrapped if side == 0 else learner_wrapped,
                    args.max_steps,
                    trace=True,
                )
                game_id = f"{run_id}:{branch['deck_id']}:{opponent_name}:{game_index:06d}:side{side}"
                decisions = finalize_trajectory(
                    learner.decisions,
                    game_id=game_id,
                    deck_id=branch["deck_id"],
                    result=int(record["result"]),
                    learner_side=side,
                    gamma=float(config["ppo"]["gamma"]),
                    gae_lambda=float(config["ppo"]["gae_lambda"]),
                )
                document = {
                    "schema_version": "top2_rl_rollout_v1",
                    "game_id": game_id,
                    "split": decisions[0]["split"] if decisions else None,
                    "role": role,
                    "candidate_id": branch["candidate_id"],
                    "deck_id": branch["deck_id"],
                    "deck_sha256": sha256_file(learner.deck_path),
                    "initial_adapter_sha256": sha256_file(learner.adapter_path),
                    "opponent": opponent_name,
                    "learner_side": side,
                    "engine_seed_controlled": False,
                    "record": record,
                    "decisions": decisions,
                }
                path = run_root / role / (document["split"] or "empty") / f"game_{opponent_name}_{game_index:06d}.json"
                write_json(path, document)
                branch_games.append(document)
        summary = {
            "role": role,
            "deck_id": branch["deck_id"],
            "games": len(branch_games),
            "decisions": sum(len(game["decisions"]) for game in branch_games),
            "exceptions": sum(len(game["record"]["exceptions"]) for game in branch_games),
            "illegal_actions": sum(sum(game["record"]["illegal_actions"]) for game in branch_games),
            "splits": {name: sum(game["split"] == name for game in branch_games) for name in ("train", "valid", "test")},
        }
        summaries.append(summary)
    report = {"schema_version": "top2_rl_rollout_summary_v1", "run_id": run_id, "root": str(run_root), "branches": summaries}
    write_json(run_root / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["exceptions"] == 0 and item["illegal_actions"] == 0 for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
