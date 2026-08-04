from __future__ import annotations

import pytest


def test_finalize_mcts_game_assigns_one_split_and_terminal_value() -> None:
    from src.rl.mcts_dataset import finalize_mcts_game

    rows = [
        {
            "step": 0,
            "actions": [[0], [1]],
            "visit_counts": [3, 1],
            "policy_target": [0.75, 0.25],
            "global_features": [0.0],
            "option_features": [[0.0], [1.0]],
            "legal_mask": [True, True],
        },
        {
            "step": 1,
            "actions": [[0], [1]],
            "visit_counts": [1, 3],
            "policy_target": [0.25, 0.75],
            "global_features": [1.0],
            "option_features": [[0.0], [1.0]],
            "legal_mask": [True, True],
        },
    ]

    result = finalize_mcts_game(
        rows,
        game_id="g-1",
        branch="primary",
        deck_id="deck-a",
        result=0,
        learner_side=0,
        checkpoint_sha256="abc",
    )

    assert {row["split"] for row in result} == {result[0]["split"]}
    assert {row["value_target"] for row in result} == {1.0}
    assert all(sum(row["policy_target"]) == pytest.approx(1.0) for row in result)


def test_schema_rejects_hidden_particle_fields() -> None:
    from src.rl.mcts_dataset import validate_mcts_sample

    with pytest.raises(ValueError, match="hidden"):
        validate_mcts_sample(
            {
                "actions": [[0]],
                "visit_counts": [1],
                "policy_target": [1.0],
                "opponent_hand": [1, 2],
            }
        )


def test_fallback_rows_are_not_valid_training_samples() -> None:
    from src.rl.mcts_dataset import validate_mcts_sample

    with pytest.raises(ValueError, match="policy target"):
        validate_mcts_sample({"actions": [], "visit_counts": [], "policy_target": []})
