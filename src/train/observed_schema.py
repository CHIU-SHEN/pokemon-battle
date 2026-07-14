"""Schema and validation for observed decisions before teacher reanalysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "observed_decision_v1"
REQUIRED_FIELDS = {
    "schema_version", "sample_id", "game_id", "source", "split",
    "step", "turn", "current_player", "select", "legal_mask",
    "observed_action", "teacher", "game_result", "value_target",
    "features", "options", "option_features", "public_history",
    "deck", "quality",
}


def split_for_game(game_id: str) -> str:
    """Return a stable 80/10/10 split without separating one game."""
    bucket = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "valid" if bucket < 90 else "test"


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
    min_count = int(select.get("min_count", 0) or 0)
    max_count = int(select.get("max_count", min_count) or min_count)
    for field in ("legal_mask", "options", "option_features"):
        if len(sample.get(field) or []) != option_count:
            errors.append(f"{field} length does not match option_count")
    action = sample.get("observed_action")
    if not isinstance(action, list):
        errors.append("observed_action is not a list")
    else:
        if not min_count <= len(action) <= max_count:
            errors.append("observed_action length outside select bounds")
        if len(action) != len(set(action)):
            errors.append("observed_action contains duplicates")
        if any(not isinstance(i, int) or i < 0 or i >= option_count for i in action):
            errors.append("observed_action contains an invalid option index")
    teacher = sample.get("teacher") or {}
    v0_action = teacher.get("v0_action")
    if v0_action is not None:
        if not isinstance(v0_action, list):
            errors.append("teacher.v0_action is not a list or null")
        elif (
            not min_count <= len(v0_action) <= max_count
            or len(v0_action) != len(set(v0_action))
            or any(not isinstance(i, int) or i < 0 or i >= option_count for i in v0_action)
        ):
            errors.append("teacher.v0_action is illegal")
    if teacher.get("v1_search") is not None:
        errors.append("teacher.v1_search must remain null until search reanalysis")
    return errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
