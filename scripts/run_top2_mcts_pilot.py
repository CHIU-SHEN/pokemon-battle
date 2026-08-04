#!/usr/bin/env python3
"""Run one resumable, non-promoting Top2 MCTS pilot branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--arena-games", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arena-device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, marker: Path, resume: bool) -> None:
    if resume and marker.is_file():
        return
    subprocess.run(command, cwd=ROOT, check=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"command": command}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve() / args.branch
    markers = root / "markers"
    samples = root / "samples"
    candidate = root / "candidate"
    python = sys.executable
    common = [
        "--config", str(args.config.resolve()),
        "--project-root", str(args.project_root.resolve()),
        "--branch", args.branch,
    ]
    run(
        [
            python, str(ROOT / "scripts/collect_top2_mcts.py"),
            *common,
            "--iteration-id", "mcts-pilot-v1",
            "--games", str(args.games),
            "--simulations", str(args.simulations),
            "--particles", str(args.particles),
            "--max-depth", str(args.max_depth),
            "--device", args.arena_device,
            "--output-root", str(samples),
            "--resume",
        ],
        marker=markers / "collect.json",
        resume=args.resume,
    )
    run(
        [
            python, str(ROOT / "scripts/train_top2_mcts.py"),
            *common,
            "--samples", str(samples),
            "--output", str(candidate),
            "--device", args.device,
            "--epochs", str(args.epochs),
        ],
        marker=markers / "train.json",
        resume=args.resume,
    )
    run(
        [
            python, str(ROOT / "scripts/evaluate_top2_mcts.py"),
            *common,
            "--kind", "search",
            "--games", str(args.arena_games),
            "--simulations", str(args.simulations),
            "--particles", str(args.particles),
            "--max-depth", str(args.max_depth),
            "--device", args.arena_device,
            "--output", str(root / "search-eval.json"),
            "--resume",
        ],
        marker=markers / "search-eval.json",
        resume=args.resume,
    )
    run(
        [
            python, str(ROOT / "scripts/evaluate_top2_mcts.py"),
            *common,
            "--kind", "candidate",
            "--checkpoint", str(candidate / "last.pt"),
            "--games", str(args.arena_games),
            "--device", args.arena_device,
            "--output", str(root / "candidate-eval.json"),
            "--resume",
        ],
        marker=markers / "candidate-eval.json",
        resume=args.resume,
    )
    report = {
        "schema_version": "top2_mcts_pilot_report_v1",
        "branch": args.branch,
        "formal_promotion_authorized": False,
        "samples": str(samples),
        "candidate": str(candidate / "last.pt"),
        "search_eval": json.loads((root / "search-eval.json").read_text(encoding="utf-8")),
        "candidate_eval": json.loads((root / "candidate-eval.json").read_text(encoding="utf-8")),
    }
    (root / "pilot-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
