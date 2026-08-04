"""Validation helpers for the first formal merged training dataset."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "training_decision_v1"
REQUIRED_FIELDS = {
    "schema_version", "sample_id", "game_id", "split", "source",
    "turn", "step", "current_player", "select", "legal_mask",
    "features", "options", "option_features", "public_history", "deck",
    "observed_action", "game_result", "value_target", "supervision",
    "quality",
}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(sample))
    if missing:
        errors.append(f"missing fields: {missing}")
    if sample.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if sample.get("split") not in {"train", "valid", "test"}:
        errors.append("invalid split")

    select = sample.get("select") or {}
    option_count = int(select.get("option_count", 0) or 0)
    minimum = int(select.get("min_count", 0) or 0)
    maximum = int(select.get("max_count", minimum) or minimum)
    for field in ("legal_mask", "options", "option_features"):
        if len(sample.get(field) or []) != option_count:
            errors.append(f"{field} length does not match option_count")

    supervision = sample.get("supervision") or {}
    target_action = supervision.get("target_action")
    if not isinstance(target_action, list):
        errors.append("supervision.target_action is not a list")
    elif (
        not minimum <= len(target_action) <= maximum
        or len(target_action) != len(set(target_action))
        or any(not isinstance(i, int) or i < 0 or i >= option_count for i in target_action)
    ):
        errors.append("supervision.target_action is illegal")

    soft_policy = supervision.get("soft_policy") or []
    if len(soft_policy) != option_count:
        errors.append("supervision.soft_policy length does not match option_count")
    elif target_action:
        if abs(sum(float(x) for x in soft_policy) - 1.0) > 1e-6:
            errors.append("non-empty target soft policy must sum to one")
    elif any(abs(float(x)) > 1e-12 for x in soft_policy):
        errors.append("empty action target must have an all-zero option policy")

    head_weights = supervision.get("head_weights") or {}
    if any(float(head_weights.get(name, -1)) < 0 for name in ("policy", "value", "risk")):
        errors.append("head weights must be non-negative")
    return errors

