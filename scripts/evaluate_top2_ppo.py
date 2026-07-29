#!/usr/bin/env python3
"""Evaluate one Top2 PPO checkpoint against its frozen initial Adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_match import play_game, summarize, with_deck  # noqa: E402
from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.arena.ppo_agent import PPOArenaAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    candidate = PPOArenaAgent(branch["candidate_id"], branch["deck_id"], args.checkpoint, project_root=args.project_root.resolve(), device=args.device)
    baseline = AdapterArenaAgent(branch["candidate_id"], project_root=args.project_root.resolve(), device=args.device)
    records = []
    for index in range(args.games):
        side = index % 2
        left = with_deck(candidate if side == 0 else baseline, candidate.deck if side == 0 else baseline.deck)
        right = with_deck(baseline if side == 0 else candidate, baseline.deck if side == 0 else candidate.deck)
        record = play_game(left, right, args.max_steps)
        if side == 1 and record["result"] in (0, 1):
            record["result"] = 1 - record["result"]
        records.append(record)
    report = summarize(records, "ppo-candidate", "initial-adapter", args.seed)
    report.update({"schema_version": "top2_ppo_arena_v1", "role": args.branch, "engine_seed_controlled": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["exceptions"] == 0 and sum(report["illegal_actions"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
