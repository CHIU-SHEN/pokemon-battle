#!/usr/bin/env python3
"""Collect resumable Top2 belief-PUCT visit targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_match import play_game, with_deck  # noqa: E402
from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.rl.belief_puct_agent import SearchConfig, Top2BeliefPUCTAgent  # noqa: E402
from src.rl.mcts_dataset import finalize_mcts_game  # noqa: E402
from src.rl.mcts_collection import audit_collection  # noqa: E402
from src.rl.top2_rollout import Top2RolloutAgent, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--iteration-id", default="mcts-pilot")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("games must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    output_root = args.output_root.resolve()
    games_root = output_root / "games"
    games_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(games_root.glob("game_*.json")) if args.resume else []
    completed = len(existing)
    if completed > args.games:
        raise ValueError("existing games exceed requested target")
    project_root = args.project_root.resolve()
    policy = Top2RolloutAgent(
        branch["candidate_id"],
        branch["deck_id"],
        project_root=project_root,
        device=args.device,
        seed=args.seed,
        deterministic=False,
        record_decisions=False,
    )
    learner = Top2BeliefPUCTAgent(
        policy,
        config=SearchConfig(
            simulations=args.simulations,
            particles=args.particles,
            max_depth=args.max_depth,
            root_noise=True,
            seed=args.seed,
        ),
        selfplay=True,
    )
    opponent = AdapterArenaAgent(
        branch["candidate_id"],
        project_root=project_root,
        device=args.device,
    )
    started = time.perf_counter()
    checkpoint_sha256 = sha256_file(policy.adapter_path)
    for game_index in range(completed, args.games):
        learner.reset_trajectory()
        side = game_index % 2
        learner_wrapped = with_deck(learner, learner.policy.deck)
        opponent_wrapped = with_deck(opponent, opponent.deck)
        record = play_game(
            learner_wrapped if side == 0 else opponent_wrapped,
            opponent_wrapped if side == 0 else learner_wrapped,
            args.max_steps,
            trace=False,
        )
        game_id = f"{args.iteration_id}:{args.branch}:{game_index:06d}:side{side}"
        samples = finalize_mcts_game(
            learner.search_records,
            game_id=game_id,
            branch=args.branch,
            deck_id=branch["deck_id"],
            result=int(record["result"]),
            learner_side=side,
            checkpoint_sha256=checkpoint_sha256,
        )
        learner_sources = {
            str(source): int(count)
            for source, count in record["action_sources"][side].items()
        }
        document = {
            "schema_version": "top2_mcts_game_v1",
            "game_id": game_id,
            "iteration_id": args.iteration_id,
            "branch": args.branch,
            "deck_id": branch["deck_id"],
            "result": record["result"],
            "learner_side": side,
            "exceptions": record["exceptions"],
            "illegal_actions": record["illegal_actions"],
            "action_sources": learner_sources,
            "samples": samples,
        }
        atomic_json(games_root / f"game_{game_index:06d}.json", document)
        completed_now = game_index + 1
        if completed_now % 10 == 0 or completed_now == args.games:
            totals = audit_collection(
                output_root,
                {"branch": args.branch, "deck_id": branch["deck_id"]},
            )
            elapsed = max(1e-9, time.perf_counter() - started)
            rate = (completed_now - completed) / elapsed
            progress = {
                "schema_version": "top2_mcts_progress_v1",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "branch": args.branch,
                "iteration_id": args.iteration_id,
                "completed_games": completed_now,
                "target_games": args.games,
                **{key: value for key, value in totals.items() if key != "game_ids"},
                "elapsed_seconds": elapsed,
                "games_per_hour": rate * 3600.0,
                "eta_seconds": (args.games - completed_now) / rate if rate > 0 else None,
            }
            atomic_json(output_root / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
