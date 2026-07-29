#!/usr/bin/env python3
"""Run the time-bounded local Top2 rollout and primary PPO pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.pilot import choose_budget_tier, select_preliminary_trial  # noqa: E402


def decode_process_output(data: bytes, encoding: str | None = None) -> str:
    return data.decode(encoding or locale.getpreferredencoding(False), errors="replace")


def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_trial_commands(
    *,
    python: str,
    root: Path,
    project_root: Path,
    rollouts: Path,
    holdout_root: Path,
    output_root: Path,
    preset: dict[str, Any],
    device: str,
    arena_games: int,
    max_wall_seconds: float,
) -> dict[str, list[str]]:
    trial_root = output_root / str(preset["name"])
    checkpoint = trial_root / "last.pt"
    common = ["--config", str(root / "config/top2_rl_policy.json"), "--project-root", str(project_root), "--branch", "primary"]
    train = [
        python, str(root / "scripts/train_top2_ppo.py"), *common,
        "--rollouts", str(rollouts), "--output", str(trial_root), "--device", device,
        "--epochs", str(preset["epochs"]), "--learning-rate", str(preset["learning_rate"]),
        "--clip-ratio", str(preset["clip_ratio"]), "--kl-coef", str(preset["kl_coef"]),
        "--entropy-coef", str(preset["entropy_coef"]), "--max-wall-seconds", str(max_wall_seconds),
    ]
    holdout = [
        python, str(root / "scripts/evaluate_top2_ppo_holdout.py"), *common,
        "--rollouts", str(holdout_root), "--checkpoint", str(checkpoint), "--device", device,
        "--output", str(trial_root / "holdout.json"),
    ]
    arena = [
        python, str(root / "scripts/evaluate_top2_ppo.py"), *common,
        "--checkpoint", str(checkpoint), "--games", str(arena_games), "--device", "cpu",
        "--output", str(trial_root / "arena.json"),
    ]
    return {"train": train, "holdout": holdout, "arena": arena}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(cwd)
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = decode_process_output(process.stdout or b"")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    print(output, end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}; log={log_path}")
    return {"command": command, "wall_seconds": elapsed, "log": str(log_path)}


def latest_run(output_root: Path) -> Path:
    runs = sorted((path for path in output_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"collector produced no run directory under {output_root}")
    return runs[-1]


def cuda_info() -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    return {
        "available": available,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if available else None,
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if available else 0,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Top2 本地 PPO Pilot 报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 运行 ID：`{report['run_id']}`",
        f"- 预算档：`{report['budget']['name']}`",
        f"- 实际总时长：{report['wall_seconds']:.2f} 秒",
        f"- 推荐组：`{report['selection'].get('selected')}`",
        "- 结论性质：preliminary，仅作为服务器正式训练起点",
        "",
        "## Rollout",
        "",
        "| 分支 | 局数 | 决策 | train | valid | test | 异常 | 非法动作 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch in report["rollout"]["branches"]:
        lines.append(
            f"| {branch['role']} | {branch['games']} | {branch['decisions']} | "
            f"{branch['splits']['train']} | {branch['splits']['valid']} | {branch['splits']['test']} | "
            f"{branch['exceptions']} | {branch['illegal_actions']} |"
        )
    lines.extend(["", "## Primary trials", "", "| 组别 | eligible | Arena | KL | value MSE | PPO 秒 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for trial in report["trials"]:
        holdout = trial.get("holdout") or {}
        lines.append(
            f"| {trial['name']} | {trial['eligible']} | {trial.get('arena_wins', 0)}/{trial.get('arena_games', 0)} | "
            f"{holdout.get('reference_kl', float('nan')):.6f} | {holdout.get('candidate_value_mse', float('nan')):.6f} | "
            f"{(trial.get('training') or {}).get('wall_seconds', 0):.2f} |"
        )
    lines.extend([
        "",
        "## 服务器下一步",
        "",
        "使用 `config/top2_rl_selected.json` 作为起点重新采集 on-policy rollout，扩大 V1 与 PPO；不得把本地 100 局结果当成正式晋级证据。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-json", type=Path, default=ROOT / "reports/top2_local_pilot_report.json")
    parser.add_argument("--report-md", type=Path, default=ROOT / "reports/top2_local_pilot_report.md")
    parser.add_argument("--selected-config", type=Path, default=ROOT / "config/top2_rl_selected.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-seconds", type=float, default=7200.0)
    parser.add_argument("--benchmark-games", type=int)
    parser.add_argument("--rollout-games", type=int)
    parser.add_argument("--arena-games", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--trial", action="append", choices=("conservative", "baseline", "exploratory"))
    parser.add_argument("--development", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot = config["pilot"]
    frozen_root = args.project_root.resolve()
    output_root = (args.output_root or (frozen_root / "experiments/adapter_top2_rl_pilot")).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / run_id
    logs_root = run_root / "logs"
    python = sys.executable
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cuda": cuda_info(),
    }
    if args.device.startswith("cuda") and not hardware["cuda"]["available"]:
        raise RuntimeError("CUDA was requested but is unavailable")

    benchmark_games = args.benchmark_games or int(pilot["benchmark_games"])
    if args.development:
        benchmark_games = 2
    benchmark_root = run_root / "benchmark"
    benchmark_command = [
        python, str(ROOT / "scripts/collect_top2_rollouts.py"), "--config", str(args.config.resolve()),
        "--project-root", str(frozen_root), "--branch", "primary", "--opponents", "cross-top2",
        "--games-per-opponent", str(benchmark_games), "--output-root", str(benchmark_root), "--device", "cpu",
    ]
    benchmark_stage = run_command(benchmark_command, cwd=ROOT, log_path=logs_root / "benchmark.log")
    per_game_seconds = benchmark_stage["wall_seconds"] / benchmark_games
    predicted_seconds = per_game_seconds * (200 + 3 * 200) + 240.0
    budget = choose_budget_tier(predicted_seconds)
    if args.development:
        budget = choose_budget_tier(0.0)
    rollout_games = args.rollout_games or budget.rollout_games_per_branch
    arena_games = args.arena_games if args.arena_games is not None else budget.arena_games
    if args.development:
        rollout_games = max(10, args.rollout_games or 10)
        arena_games = args.arena_games if args.arena_games is not None else 2

    rollout_root = run_root / "rollouts"
    rollout_command = [
        python, str(ROOT / "scripts/collect_top2_rollouts.py"), "--config", str(args.config.resolve()),
        "--project-root", str(frozen_root), "--branch", "all", "--opponents", "cross-top2",
        "--games-per-opponent", str(rollout_games), "--output-root", str(rollout_root), "--device", "cpu",
    ]
    rollout_stage = run_command(rollout_command, cwd=ROOT, log_path=logs_root / "rollout.log")
    rollout_run = latest_run(rollout_root)
    rollout_summary = json.loads((rollout_run / "summary.json").read_text(encoding="utf-8"))
    deck_ids = {item["deck_id"] for item in rollout_summary["branches"]}
    if len(deck_ids) != 2:
        raise RuntimeError("primary/reserve rollout deck_id streams are not distinct")
    for branch in rollout_summary["branches"]:
        if branch["games"] != rollout_games or branch["exceptions"] or branch["illegal_actions"]:
            raise RuntimeError(f"rollout hard gate failed: {branch}")
        if not branch["splits"]["train"] or not (branch["splits"]["valid"] + branch["splits"]["test"]):
            raise RuntimeError(f"rollout lacks train or holdout games: {branch}")

    preset_by_name = {item["name"]: item for item in pilot["presets"]}
    trial_names = args.trial or list(budget.trials)
    if args.development and args.trial is None:
        trial_names = ["conservative"]
    trials = []
    trial_output = run_root / "trials"
    primary_root = rollout_run / "primary"
    primary_train = primary_root / "train"
    for index, name in enumerate(trial_names):
        if time.perf_counter() - started >= args.max_wall_seconds:
            break
        preset = dict(preset_by_name[name])
        if budget.epoch_cap is not None:
            preset["epochs"] = min(int(preset["epochs"]), budget.epoch_cap)
        if args.development:
            preset["epochs"] = 1
        remaining = max(1.0, args.max_wall_seconds - (time.perf_counter() - started))
        commands = build_trial_commands(
            python=python, root=ROOT, project_root=frozen_root, rollouts=primary_train,
            holdout_root=primary_root, output_root=trial_output, preset=preset, device=args.device,
            arena_games=arena_games, max_wall_seconds=remaining / max(1, len(trial_names) - index),
        )
        max_batches = 1 if args.development and not args.max_batches else args.max_batches
        if max_batches:
            commands["train"].extend(["--max-batches", str(max_batches)])
        train_stage = run_command(commands["train"], cwd=ROOT, log_path=logs_root / f"{name}_train.log")
        train_summary = json.loads((trial_output / name / "summary.json").read_text(encoding="utf-8"))
        trial = {"name": name, "preset": preset, "training": train_summary, "stages": {"train": train_stage}}
        eligible = bool(train_summary["eligible"])
        if eligible:
            holdout_stage = run_command(commands["holdout"], cwd=ROOT, log_path=logs_root / f"{name}_holdout.log")
            holdout = json.loads((trial_output / name / "holdout.json").read_text(encoding="utf-8"))
            trial["holdout"] = holdout
            trial["stages"]["holdout"] = holdout_stage
            eligible = (
                holdout["finite"] and holdout["illegal_argmax"] == 0 and holdout["reference_kl"] <= 0.03
                and holdout["candidate_action_accuracy"] + 0.05 >= holdout["reference_action_accuracy"]
                and holdout["candidate_value_mse"] <= holdout["reference_value_mse"] * 1.10 + 0.01
            )
        if eligible and arena_games > 0:
            arena_stage = run_command(commands["arena"], cwd=ROOT, log_path=logs_root / f"{name}_arena.log")
            arena = json.loads((trial_output / name / "arena.json").read_text(encoding="utf-8"))
            trial["arena"] = arena
            trial["stages"]["arena"] = arena_stage
            trial["arena_wins"] = int(arena["agent0_wins"])
            trial["arena_games"] = int(arena["games"])
            eligible = arena["exceptions"] == 0 and sum(arena["illegal_actions"]) == 0
        else:
            trial["arena_wins"] = 0
            trial["arena_games"] = 0
        trial["eligible"] = bool(eligible)
        trials.append(trial)

    if arena_games > 0:
        selection = select_preliminary_trial(trials)
    else:
        safe_names = {trial["name"] for trial in trials if trial["eligible"]}
        selected = next((name for name in ("conservative", "baseline") if name in safe_names), None)
        selection = {"status": "minimal_no_arena", "selected": selected, "eligible": [trial for trial in trials if trial["eligible"]]}
    selected_name = selection.get("selected")
    selected_preset = preset_by_name.get(selected_name) if selected_name else None
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "top2_local_pilot_report_v1",
        "status": "completed_preliminary" if selected_name else "completed_without_recommendation",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed,
        "max_wall_seconds": args.max_wall_seconds,
        "hardware": hardware,
        "frozen_source_root": str(frozen_root),
        "frozen_hashes": {
            "shared_checkpoint": config["shared_checkpoint"]["sha256"],
            **{f"{item['role']}_deck": item["deck_sha256"] for item in config["branches"]},
            **{f"{item['role']}_adapter": item["adapter_sha256"] for item in config["branches"]},
        },
        "benchmark": {**benchmark_stage, "games": benchmark_games, "seconds_per_game": per_game_seconds, "predicted_full_seconds": predicted_seconds},
        "budget": {
            "name": budget.name, "rollout_games_per_branch": rollout_games,
            "arena_games_per_trial": arena_games, "trials": trial_names, "epoch_cap": budget.epoch_cap,
        },
        "rollout": {
            **rollout_stage,
            "run_root": str(rollout_run),
            "branches": rollout_summary["branches"],
            "data_bytes": tree_bytes(rollout_run),
            "bytes_per_game": tree_bytes(rollout_run) / max(1, sum(item["games"] for item in rollout_summary["branches"])),
        },
        "trials": trials,
        "selection": selection,
        "preliminary_only": True,
        "submission_replacement_authorized": False,
        "remaining_server_work": ["large on-policy rollout", "Top2 V1 expansion", "full masked PPO", "frozen holdout and final Arena"],
    }
    selected_config = {
        "schema_version": "top2_rl_selected_v1",
        "status": selection["status"],
        "selected_preset": selected_name,
        "parameters": selected_preset,
        "source_report": "reports/top2_local_pilot_report.json",
        "source_run_id": run_id,
        "frozen_hashes": report["frozen_hashes"],
        "preliminary_only": True,
        "submission_replacement_authorized": False,
    }
    for path, value in ((args.report_json, report), (args.selected_config, selected_config)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"report": str(args.report_json), "selected": selected_name, "status": selection["status"], "wall_seconds": elapsed}, ensure_ascii=False, indent=2))
    return 0 if selected_name else 1


if __name__ == "__main__":
    raise SystemExit(main())
