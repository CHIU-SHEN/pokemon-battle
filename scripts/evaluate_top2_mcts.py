#!/usr/bin/env python3
"""Evaluate best+MCTS or a distilled MCTS candidate against pure best."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_match import play_game, with_deck  # noqa: E402
from src.rl.belief_puct_agent import SearchConfig, Top2BeliefPUCTAgent  # noqa: E402
from src.rl.mcts_gate import mcts_gate_decision  # noqa: E402
from src.rl.top2_rollout import Top2RolloutAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--kind", choices=("search", "candidate"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--time-budget-seconds", type=float, default=0.030)
    parser.add_argument("--game-budget-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluation_identity(
    *,
    branch: str,
    kind: str,
    games: int,
    simulations: int,
    particles: int,
    max_depth: int,
    checkpoint: Path | None,
    time_budget_seconds: float = 0.030,
    game_budget_seconds: float = 2.0,
) -> dict:
    return {
        "branch": branch,
        "kind": kind,
        "games": games,
        "simulations": simulations,
        "particles": particles,
        "max_depth": max_depth,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "time_budget_seconds": time_budget_seconds,
        "game_budget_seconds": game_budget_seconds,
    }


def new_progress(identity: dict) -> dict:
    return {
        "schema_version": "top2_mcts_eval_progress_v1",
        "identity": identity,
        "completed_games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "exceptions": 0,
        "illegal_actions": 0,
        "fallbacks": 0,
        "decisions": 0,
        "latencies": [],
        "tested_sides": [],
        "action_sources": {},
    }


def load_progress(path: Path, identity: dict, *, resume: bool) -> dict:
    if not resume or not path.is_file():
        return new_progress(identity)
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("identity") != identity:
        raise ValueError("saved evaluation progress does not match requested run")
    completed = int(progress.get("completed_games", -1))
    if completed < 0 or completed > int(identity["games"]):
        raise ValueError("saved evaluation progress has invalid completed game count")
    if len(progress.get("tested_sides", [])) != completed:
        raise ValueError("saved evaluation progress has inconsistent side history")
    return progress


def record_completed_game(
    path: Path,
    progress: dict,
    *,
    game_index: int,
    result: int,
    tested_side: int,
    exceptions: int,
    illegal_actions: int,
    fallbacks: int,
    decisions: int,
    latencies: list[float],
    action_sources: dict[str, int] | None = None,
) -> None:
    if game_index != progress["completed_games"]:
        raise ValueError("game index is not the next resumable game")
    if result == 2:
        progress["draws"] += 1
    elif result == tested_side:
        progress["wins"] += 1
    else:
        progress["losses"] += 1
    progress["exceptions"] += exceptions
    progress["illegal_actions"] += illegal_actions
    progress["fallbacks"] += fallbacks
    progress["decisions"] += decisions
    progress["latencies"].extend(float(value) for value in latencies)
    progress["tested_sides"].append(tested_side)
    totals = progress.setdefault("action_sources", {})
    for source, count in (action_sources or {}).items():
        totals[source] = int(totals.get(source, 0)) + int(count)
    progress["completed_games"] = game_index + 1
    atomic_write_json(path, progress)


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("games must be positive")
    if args.kind == "candidate" and not args.checkpoint:
        raise ValueError("candidate evaluation requires --checkpoint")
    identity = evaluation_identity(
        branch=args.branch,
        kind=args.kind,
        games=args.games,
        simulations=args.simulations,
        particles=args.particles,
        max_depth=args.max_depth,
        checkpoint=args.checkpoint,
        time_budget_seconds=args.time_budget_seconds,
        game_budget_seconds=args.game_budget_seconds,
    )
    output = args.output.resolve()
    progress_path = output.with_suffix(".progress.json")
    progress = load_progress(progress_path, identity, resume=args.resume)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    project_root = args.project_root.resolve()
    baseline = Top2RolloutAgent(
        branch["candidate_id"],
        branch["deck_id"],
        project_root=project_root,
        device=args.device,
        deterministic=True,
        record_decisions=False,
    )
    tested_policy = Top2RolloutAgent(
        branch["candidate_id"],
        branch["deck_id"],
        project_root=project_root,
        device=args.device,
        deterministic=True,
        record_decisions=False,
        ppo_checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
    )
    tested = (
        Top2BeliefPUCTAgent(
            tested_policy,
            config=SearchConfig(
                simulations=args.simulations,
                particles=args.particles,
                max_depth=args.max_depth,
                root_noise=False,
                time_budget_seconds=args.time_budget_seconds,
                game_budget_seconds=args.game_budget_seconds,
            ),
            selfplay=False,
        )
        if args.kind == "search"
        else tested_policy
    )
    for game_index in range(progress["completed_games"], args.games):
        if hasattr(tested, "reset_trajectory"):
            tested.reset_trajectory()
        tested_side = game_index % 2
        tested_wrapped = with_deck(tested, tested_policy.deck)
        baseline_wrapped = with_deck(baseline, baseline.deck)
        record = play_game(
            tested_wrapped if tested_side == 0 else baseline_wrapped,
            baseline_wrapped if tested_side == 0 else tested_wrapped,
            1000,
            trace=False,
        )
        action_sources = record["action_sources"][tested_side]
        record_completed_game(
            progress_path,
            progress,
            game_index=game_index,
            result=int(record["result"]),
            tested_side=tested_side,
            exceptions=len(record["exceptions"]),
            illegal_actions=sum(record["illegal_actions"]),
            fallbacks=int(action_sources.get("mcts_fallback", 0)),
            decisions=sum(int(value) for value in action_sources.values()),
            latencies=record["agent_decision_times"][tested_side],
            action_sources=action_sources,
        )
        print(
            json.dumps({
                "branch": args.branch,
                "kind": args.kind,
                "completed_games": progress["completed_games"],
                "target_games": args.games,
            }),
            flush=True,
        )
    wins = progress["wins"]
    losses = progress["losses"]
    draws = progress["draws"]
    exceptions = progress["exceptions"]
    illegal = progress["illegal_actions"]
    fallbacks = progress["fallbacks"]
    decisions = progress["decisions"]
    latencies = progress["latencies"]
    fallback_rate = fallbacks / max(1, decisions)
    gate = mcts_gate_decision(
        wins,
        losses,
        draws,
        kind=args.kind,
        exceptions=exceptions,
        illegal_actions=illegal,
        fallback_rate=fallback_rate,
    )
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0
    report = {
        "schema_version": "top2_mcts_eval_v1",
        "branch": args.branch,
        "kind": args.kind,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "exceptions": exceptions,
        "illegal_actions": illegal,
        "fallback_rate": fallback_rate,
        "mean_decision_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "p95_decision_seconds": p95,
        "action_sources": progress.get("action_sources", {}),
        "gate": gate.__dict__,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if exceptions == 0 and illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
