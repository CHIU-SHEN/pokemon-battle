#!/usr/bin/env python3
"""Evaluate best+MCTS or a distilled MCTS candidate against pure best."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.kind == "candidate" and not args.checkpoint:
        raise ValueError("candidate evaluation requires --checkpoint")
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
            ),
            selfplay=False,
        )
        if args.kind == "search"
        else tested_policy
    )
    wins = losses = draws = exceptions = illegal = fallbacks = decisions = 0
    latencies = []
    for game_index in range(args.games):
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
        if record["result"] == 2:
            draws += 1
        elif record["result"] == tested_side:
            wins += 1
        else:
            losses += 1
        exceptions += len(record["exceptions"])
        illegal += sum(record["illegal_actions"])
        action_sources = record["action_sources"][tested_side]
        fallbacks += int(action_sources.get("mcts_fallback", 0))
        decisions += sum(int(value) for value in action_sources.values())
        latencies.extend(record["agent_decision_times"][tested_side])
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
        "gate": gate.__dict__,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if exceptions == 0 and illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
