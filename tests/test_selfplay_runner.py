from __future__ import annotations

from pathlib import Path


def checkpoint(root: Path, name: str) -> Path:
    path = root / f"{name}.pt"
    path.write_bytes(name.encode("utf-8"))
    return path


def initialized_state(tmp_path: Path):
    from src.rl.selfplay_state import SelfPlayState

    return SelfPlayState.load_or_initialize(
        tmp_path / "primary",
        "primary",
        "deck-primary",
        checkpoint(tmp_path, "initial"),
    )


def test_resume_skips_completed_rollout(tmp_path: Path) -> None:
    from src.rl.selfplay_runner import SelfPlayRunner

    calls = {"rollout": 0, "train": 0}

    def rollout(context):
        calls["rollout"] += 1
        return {"games": 10}

    def train(context):
        calls["train"] += 1
        return {"checkpoint": str(checkpoint(tmp_path, "candidate"))}

    state = initialized_state(tmp_path)
    runner = SelfPlayRunner(
        state,
        "iter-0001",
        stages={"rollout": rollout, "train": train},
    )
    runner.run(stop_after="rollout")
    runner.run()

    assert calls == {"rollout": 1, "train": 1}


def test_rejected_candidate_does_not_change_best(tmp_path: Path) -> None:
    from src.rl.selfplay_runner import SelfPlayRunner

    state = initialized_state(tmp_path)
    before = state.best["sha256"]
    candidate = checkpoint(tmp_path, "candidate")
    runner = SelfPlayRunner(
        state,
        "iter-0001",
        stages={
            "rollout": lambda context: {"games": 10},
            "train": lambda context: {"checkpoint": str(candidate)},
            "gate": lambda context: {
                "status": "reject",
                "reason": "initial_rate_at_most_52_percent",
                "win_rate": 0.49,
            },
        },
    )
    report = runner.run()

    assert report["status"] == "rejected"
    assert state.best["sha256"] == before


def test_promotion_uses_trained_candidate_only_after_regression_passes(tmp_path: Path) -> None:
    from src.rl.selfplay_runner import SelfPlayRunner

    state = initialized_state(tmp_path)
    candidate = checkpoint(tmp_path, "candidate")
    runner = SelfPlayRunner(
        state,
        "iter-0001",
        stages={
            "rollout": lambda context: {"games": 10},
            "train": lambda context: {"checkpoint": str(candidate)},
            "gate": lambda context: {"status": "promote_ready", "win_rate": 0.60},
            "regression": lambda context: {"passed": True, "illegal_actions": 0},
        },
    )
    report = runner.run()

    assert report["status"] == "promoted"
    assert state.best["source_iteration"] == "iter-0001"
