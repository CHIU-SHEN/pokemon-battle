"""Schema helpers for public belief-PUCT training targets."""

from __future__ import annotations

from typing import Any

from src.rl.top2_rollout import stable_game_split


FORBIDDEN_HIDDEN_KEYS = {
    "your_deck_hidden",
    "your_prize",
    "opponent_deck",
    "opponent_prize",
    "opponent_hand",
    "opponent_active",
    "particle",
    "particles",
}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def validate_mcts_sample(sample: dict[str, Any]) -> dict[str, Any]:
    hidden = _keys(sample) & FORBIDDEN_HIDDEN_KEYS
    if hidden:
        raise ValueError(f"hidden belief fields cannot be serialized: {sorted(hidden)}")
    actions = sample.get("actions") or []
    visits = sample.get("visit_counts") or []
    target = sample.get("policy_target") or []
    if not actions or len(actions) != len(visits) or len(actions) != len(target):
        raise ValueError("valid MCTS policy target is required")
    if any(int(count) < 0 for count in visits) or sum(int(count) for count in visits) <= 0:
        raise ValueError("visit counts must have positive mass")
    mass = sum(float(value) for value in target)
    if abs(mass - 1.0) > 1e-6:
        raise ValueError("policy target must sum to one")
    return sample


def finalize_mcts_game(
    records: list[dict[str, Any]],
    *,
    game_id: str,
    branch: str,
    deck_id: str,
    result: int,
    learner_side: int,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    value_target = 0.0 if result == 2 else 1.0 if result == learner_side else -1.0
    split = stable_game_split(game_id)
    finalized = []
    for source in records:
        row = dict(source)
        row.update(
            {
                "schema_version": "top2_mcts_sample_v1",
                "game_id": game_id,
                "branch": branch,
                "deck_id": deck_id,
                "split": split,
                "best_checkpoint_sha256": checkpoint_sha256,
                "value_target": value_target,
            }
        )
        finalized.append(validate_mcts_sample(row))
    return finalized
