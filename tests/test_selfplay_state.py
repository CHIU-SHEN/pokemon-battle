from __future__ import annotations

import json
from pathlib import Path

import pytest


def checkpoint(root: Path, name: str, payload: bytes | None = None) -> Path:
    path = root / f"{name}.pt"
    path.write_bytes(payload or name.encode("utf-8"))
    return path


def test_primary_and_reserve_roots_cannot_cross(tmp_path: Path) -> None:
    from src.rl.selfplay_state import SelfPlayState

    primary = SelfPlayState.load_or_initialize(
        tmp_path / "primary",
        "primary",
        "deck-primary",
        checkpoint(tmp_path, "primary-initial"),
    )

    with pytest.raises(ValueError, match="branch"):
        SelfPlayState.load(primary.root, expected_branch="reserve")


def test_promote_archives_old_best_before_switching_pointer(tmp_path: Path) -> None:
    from src.rl.selfplay_state import SelfPlayState

    state = SelfPlayState.load_or_initialize(
        tmp_path / "primary",
        "primary",
        "deck-primary",
        checkpoint(tmp_path, "initial"),
    )
    old_sha = state.best["sha256"]
    assert state.best["branch"] == "primary"
    assert state.best["deck_id"] == "deck-primary"
    assert state.best["checkpoint_kind"] == "adapter"
    state.begin_iteration("iter-0001")
    state.promote(checkpoint(tmp_path, "candidate"), {"win_rate": 0.60})

    assert state.history[-1]["sha256"] == old_sha
    assert state.best["sha256"] != old_sha
    assert state.best["checkpoint_kind"] == "ppo"
    assert Path(state.history[-1]["path"]).is_file()
    assert state.iteration["status"] == "promoted"


def test_reject_keeps_best_and_records_reason(tmp_path: Path) -> None:
    from src.rl.selfplay_state import SelfPlayState

    state = SelfPlayState.load_or_initialize(
        tmp_path / "reserve",
        "reserve",
        "deck-reserve",
        checkpoint(tmp_path, "initial"),
    )
    old_best = dict(state.best)
    state.begin_iteration("iter-0001")
    state.reject("arena_gate", {"win_rate": 0.49})

    assert state.best == old_best
    assert state.iteration["status"] == "rejected"
    assert state.iteration["reason"] == "arena_gate"


def test_repeated_stage_completion_is_idempotent(tmp_path: Path) -> None:
    from src.rl.selfplay_state import SelfPlayState

    state = SelfPlayState.load_or_initialize(
        tmp_path / "primary",
        "primary",
        "deck-primary",
        checkpoint(tmp_path, "initial"),
    )
    state.begin_iteration("iter-0001")
    artifacts = {"manifest": "iterations/iter-0001/rollout/manifest.json", "games": 10}
    state.complete_stage("rollout", artifacts)
    first = json.loads(state.path.read_text(encoding="utf-8"))
    state.complete_stage("rollout", artifacts)
    second = json.loads(state.path.read_text(encoding="utf-8"))

    assert first == second


def test_stage_cannot_be_rewritten_with_different_artifacts(tmp_path: Path) -> None:
    from src.rl.selfplay_state import SelfPlayState

    state = SelfPlayState.load_or_initialize(
        tmp_path / "primary",
        "primary",
        "deck-primary",
        checkpoint(tmp_path, "initial"),
    )
    state.begin_iteration("iter-0001")
    state.complete_stage("rollout", {"games": 10})

    with pytest.raises(ValueError, match="already completed"):
        state.complete_stage("rollout", {"games": 11})
