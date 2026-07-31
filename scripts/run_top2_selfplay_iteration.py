#!/usr/bin/env python3
"""Run or resume one branch-bound gated Top2 self-play iteration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.selfplay_gate import gate_decision  # noqa: E402
from src.rl.selfplay_runner import SelfPlayRunner  # noqa: E402
from src.rl.selfplay_state import SelfPlayState  # noqa: E402


def newest_child(path: Path) -> Path:
    children = [item for item in path.iterdir() if item.is_dir()]
    if not children:
        raise FileNotFoundError(f"no run directory under {path}")
    return max(children, key=lambda item: item.stat().st_mtime_ns)


def run_command(command: list[str], *, cwd: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cwd)
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--selfplay-root", type=Path, required=True)
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--rollout-games", type=int, default=3000)
    parser.add_argument("--gate-games", type=int, default=1000)
    parser.add_argument("--gate-cap", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arena-device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    project_root = args.project_root.resolve()
    state = SelfPlayState.load_or_initialize(
        args.selfplay_root.resolve() / args.branch,
        args.branch,
        branch["deck_id"],
        project_root / branch["adapter_path"],
    )
    iteration_root = state.root / "iterations" / args.iteration_id
    python = sys.executable

    def rollout_stage(context: dict[str, Any]) -> dict[str, Any]:
        output = iteration_root / "rollout"
        command = [
            python,
            str(ROOT / "scripts/collect_top2_rollouts.py"),
            "--config",
            str(args.config.resolve()),
            "--project-root",
            str(project_root),
            "--branch",
            args.branch,
            "--selfplay-root",
            str(args.selfplay_root.resolve()),
            "--iteration-id",
            args.iteration_id,
            "--games",
            str(args.rollout_games),
            "--output-root",
            str(output),
            "--device",
            "cpu",
        ]
        run_command(command, cwd=ROOT)
        run_root = newest_child(output)
        summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        return {
            "run_root": str(run_root),
            "summary": str(run_root / "summary.json"),
            "games": summary["branches"][0]["games"],
        }

    def train_stage(context: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(context["stages"]["rollout"]["run_root"])
        output = iteration_root / "candidate"
        command = [
            python,
            str(ROOT / "scripts/train_top2_ppo.py"),
            "--config",
            str(args.config.resolve()),
            "--project-root",
            str(project_root),
            "--branch",
            args.branch,
            "--rollouts",
            str(run_root / args.branch / "train"),
            "--output",
            str(output),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
        ]
        if context["best"].get("checkpoint_kind") == "ppo":
            command.extend(["--initial-checkpoint", context["best"]["path"]])
        run_command(command, cwd=ROOT)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if not summary["eligible"]:
            raise RuntimeError(f"candidate failed PPO safety gates: {summary['stop_reason']}")
        return {"checkpoint": str(output / "last.pt"), "summary": str(output / "summary.json")}

    def holdout_stage(context: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(context["stages"]["rollout"]["run_root"])
        output = iteration_root / "holdout.json"
        command = [
            python,
            str(ROOT / "scripts/evaluate_top2_ppo_holdout.py"),
            "--config",
            str(args.config.resolve()),
            "--project-root",
            str(project_root),
            "--branch",
            args.branch,
            "--rollouts",
            str(run_root / args.branch),
            "--checkpoint",
            context["stages"]["train"]["checkpoint"],
            "--device",
            args.device,
            "--output",
            str(output),
        ]
        if context["best"].get("checkpoint_kind") == "ppo":
            command.extend(["--reference-checkpoint", context["best"]["path"]])
        run_command(command, cwd=ROOT)
        report = json.loads(output.read_text(encoding="utf-8"))
        return {"report": str(output), **report}

    def gate_stage(context: dict[str, Any]) -> dict[str, Any]:
        total = {"wins": 0, "losses": 0, "draws": 0}
        batch = args.gate_games
        while total["wins"] + total["losses"] + total["draws"] < args.gate_cap:
            output = iteration_root / "arena" / f"batch-{sum(total.values()):04d}.json"
            command = [
                python,
                str(ROOT / "scripts/evaluate_top2_ppo.py"),
                "--config",
                str(args.config.resolve()),
                "--project-root",
                str(project_root),
                "--branch",
                args.branch,
                "--checkpoint",
                context["stages"]["train"]["checkpoint"],
                "--games",
                str(min(batch, args.gate_cap - sum(total.values()))),
                "--device",
                args.arena_device,
                "--output",
                str(output),
            ]
            if context["best"].get("checkpoint_kind") == "ppo":
                command.extend(["--baseline-checkpoint", context["best"]["path"]])
            run_command(command, cwd=ROOT)
            report = json.loads(output.read_text(encoding="utf-8"))
            total["wins"] += int(report["agent0_wins"])
            total["losses"] += int(report["agent1_wins"])
            total["draws"] += int(report["draws"])
            if args.gate_cap < 1000:
                return {
                    "status": "reject",
                    "reason": "smoke_only_not_promotion_eligible",
                    **total,
                }
            decision = gate_decision(
                total["wins"],
                total["losses"],
                total["draws"],
                games_cap=args.gate_cap,
            )
            if decision.status != "continue":
                return decision.to_dict()
        return gate_decision(
            total["wins"],
            total["losses"],
            total["draws"],
            games_cap=args.gate_cap,
        ).to_dict()

    def regression_stage(context: dict[str, Any]) -> dict[str, Any]:
        holdout = context["stages"]["holdout"]
        return {
            "passed": bool(holdout.get("finite", False)) and int(holdout.get("illegal_argmax", 1)) == 0,
            "illegal_actions": int(holdout.get("illegal_argmax", 1)),
            "holdout_report": holdout["report"],
        }

    runner = SelfPlayRunner(
        state,
        args.iteration_id,
        stages={
            "rollout": rollout_stage,
            "train": train_stage,
            "holdout": holdout_stage,
            "gate": gate_stage,
            "regression": regression_stage,
        },
    )
    report = runner.run()
    report_path = iteration_root / "iteration-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
