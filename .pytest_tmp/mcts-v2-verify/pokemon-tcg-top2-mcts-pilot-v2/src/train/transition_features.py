"""Leakage-safe temporal features between two same-perspective decisions."""

from __future__ import annotations

from typing import Any


# The first 22 global features describe the public game state.  Selection
# context starts at index 22 and must not be differenced across decisions.
DYNAMIC_GLOBAL_NAMES = (
    "turn",
    "turn_action_count",
    "supporter_played",
    "energy_attached",
    "own_prize_remaining",
    "opp_prize_remaining",
    "own_hand_count",
    "opp_hand_count",
    "own_deck_count",
    "opp_deck_count",
    "own_bench_count",
    "opp_bench_count",
    "own_active_hp",
    "own_active_max_hp",
    "own_active_damage",
    "own_active_energy",
    "opp_active_hp",
    "opp_active_max_hp",
    "opp_active_damage",
    "opp_active_energy",
    "own_discard_count",
    "own_discard_water_energy",
)
TRANSITION_FEATURE_NAMES = tuple(f"delta_{name}" for name in DYNAMIC_GLOBAL_NAMES) + (
    "turn_changed",
    "visible_log_present",
)
TRANSITION_DIM = len(TRANSITION_FEATURE_NAMES)


def transition_features(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[float]:
    """Return only information visible at ``current`` from the same player view."""
    if previous is None:
        return [0.0] * TRANSITION_DIM
    before = previous.get("features") or []
    after = current.get("features") or []
    if len(before) < len(DYNAMIC_GLOBAL_NAMES) or len(after) < len(DYNAMIC_GLOBAL_NAMES):
        raise ValueError("global feature vector is too short for transition features")
    deltas = [
        max(-1.0, min(1.0, float(after[index]) - float(before[index])))
        for index in range(len(DYNAMIC_GLOBAL_NAMES))
    ]
    turn_changed = float(int(current.get("turn", 0)) != int(previous.get("turn", 0)))
    visible_log_present = float(bool(current.get("public_history")))
    return deltas + [turn_changed, visible_log_present]


def previous_action_features(previous: dict[str, Any] | None, option_dim: int) -> list[float]:
    """Mean feature vector of the previous action selected by this player."""
    if previous is None:
        return [0.0] * option_dim
    options = previous.get("option_features") or []
    chosen = [
        options[int(index)]
        for index in previous.get("observed_action") or []
        if 0 <= int(index) < len(options)
    ]
    if not chosen:
        return [0.0] * option_dim
    if any(len(option) != option_dim for option in chosen):
        raise ValueError("inconsistent previous action feature dimension")
    return [sum(float(option[index]) for option in chosen) / len(chosen) for index in range(option_dim)]
