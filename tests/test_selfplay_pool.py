from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch


def policy(path: Path, name: str, *, branch: str = "primary", deck_id: str = "deck-primary") -> dict:
    checkpoint = path / f"{name}.pt"
    torch.save(
        {
            "schema_version": "top2_ppo_checkpoint_v1",
            "candidate_id": f"candidate-{branch}",
            "deck_id": deck_id,
            "adapter_state": {},
        },
        checkpoint,
    )
    return {
        "kind": name,
        "path": str(checkpoint),
        "sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "branch": branch,
        "deck_id": deck_id,
    }


def test_empty_history_weight_falls_back_to_best(tmp_path: Path) -> None:
    from src.rl.selfplay_pool import build_opponent_schedule

    best = policy(tmp_path, "best")
    schedule = build_opponent_schedule(
        best=best,
        history=[],
        games=100,
        seed=7,
        baselines=("random", "first-min"),
    )
    kinds = Counter(item.kind for item in schedule)

    assert kinds["best"] == 80
    assert kinds["baseline"] == 20


def test_history_receives_exact_thirty_percent_and_schedule_is_deterministic(tmp_path: Path) -> None:
    from src.rl.selfplay_pool import build_opponent_schedule

    best = policy(tmp_path, "best")
    history = [policy(tmp_path, "history-1"), policy(tmp_path, "history-2")]
    first = build_opponent_schedule(best=best, history=history, games=100, seed=11)
    second = build_opponent_schedule(best=best, history=history, games=100, seed=11)
    kinds = Counter(item.kind for item in first)

    assert kinds == {"best": 50, "history": 30, "baseline": 20}
    assert first == second


def test_checkpoint_identity_rejects_other_branch(tmp_path: Path) -> None:
    from src.rl.selfplay_pool import validate_checkpoint_identity

    reserve = policy(
        tmp_path,
        "reserve",
        branch="reserve",
        deck_id="deck-reserve",
    )

    with pytest.raises(ValueError, match="deck_id"):
        validate_checkpoint_identity(
            Path(reserve["path"]),
            expected_candidate_id="candidate-primary",
            expected_deck_id="deck-primary",
        )


def test_game_manifest_records_iteration_and_both_policy_hashes(tmp_path: Path) -> None:
    from src.rl.selfplay_pool import build_game_identity

    learner = policy(tmp_path, "learner")
    opponent = policy(tmp_path, "opponent")
    row = build_game_identity(
        iteration_id="iter-0001",
        game_index=4,
        learner=learner,
        opponent=opponent,
    )

    assert row["iteration_id"] == "iter-0001"
    assert row["learner_sha256"] == learner["sha256"]
    assert row["opponent_sha256"] == opponent["sha256"]
