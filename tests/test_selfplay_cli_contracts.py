from __future__ import annotations

from pathlib import Path
import sys


def test_rollout_cli_accepts_selfplay_state_and_iteration(monkeypatch) -> None:
    from scripts.collect_top2_rollouts import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_top2_rollouts.py",
            "--branch",
            "primary",
            "--selfplay-root",
            "selfplay",
            "--iteration-id",
            "iter-0001",
            "--games",
            "3000",
        ],
    )
    args = parse_args()

    assert args.selfplay_root == Path("selfplay")
    assert args.iteration_id == "iter-0001"
    assert args.games == 3000


def test_ppo_cli_accepts_parent_best_checkpoint(monkeypatch) -> None:
    from scripts.train_top2_ppo import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_top2_ppo.py",
            "--branch",
            "primary",
            "--rollouts",
            "rollouts",
            "--output",
            "candidate",
            "--initial-checkpoint",
            "best.pt",
        ],
    )
    args = parse_args()

    assert args.initial_checkpoint == Path("best.pt")


def test_iteration_cli_defaults_to_formal_selfplay_budgets(monkeypatch) -> None:
    from scripts.run_top2_selfplay_iteration import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_top2_selfplay_iteration.py",
            "--branch",
            "reserve",
            "--selfplay-root",
            "selfplay",
            "--iteration-id",
            "iter-0001",
        ],
    )
    args = parse_args()

    assert args.rollout_games == 3000
    assert args.gate_games == 1000
    assert args.gate_cap == 3000
