"""Fixed schema and validation helpers for M6 distillation samples."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "m6_distill_v1"

REQUIRED_FIELDS = {
    "schema_version",
    "sample_id",
    "game_id",
    "turn",
    "current_player",
    "select",
    "legal_mask",
    "v0_action",
    "search_action",
    "final_action",
    "search",
    "game_result",
    "budget",
    "is_key",
    "features",
    "options",
    "option_features",
    "value_target",
    "risk_target",
}


@dataclass(frozen=True)
class SampleValidation:
    ok: bool
    errors: list[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_sample(sample: dict[str, Any]) -> SampleValidation:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(sample))
    if missing:
        errors.append(f"missing fields: {missing}")
    if sample.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    select = sample.get("select") or {}
    option_count = int(select.get("option_count", 0) or 0)
    min_count = int(select.get("min_count", 0) or 0)
    max_count = int(select.get("max_count", min_count) or min_count)

    legal_mask = sample.get("legal_mask") or []
    if len(legal_mask) != option_count:
        errors.append("legal_mask length does not match option_count")
    options = sample.get("options") or []
    option_features = sample.get("option_features") or []
    if len(options) != option_count:
        errors.append("options length does not match option_count")
    if len(option_features) != option_count:
        errors.append("option_features length does not match option_count")

    for field in ("v0_action", "search_action", "final_action"):
        action = sample.get(field)
        if not isinstance(action, list):
            errors.append(f"{field} is not a list")
            continue
        if len(action) < min_count or len(action) > max_count:
            errors.append(f"{field} length outside select bounds")
        if len(set(action)) != len(action):
            errors.append(f"{field} has duplicate option indices")
        bad = [idx for idx in action if not isinstance(idx, int) or idx < 0 or idx >= option_count]
        if bad:
            errors.append(f"{field} has out-of-range indices: {bad}")

    return SampleValidation(ok=not errors, errors=errors)


def validate_dataset(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    errors: list[dict[str, Any]] = []
    game_ids: set[str] = set()
    for sample in samples:
        count += 1
        game_ids.add(str(sample.get("game_id")))
        result = validate_sample(sample)
        if not result.ok:
            errors.append({"sample_id": sample.get("sample_id"), "errors": result.errors})
    return {
        "ok": not errors and count > 0,
        "sample_count": count,
        "game_count": len(game_ids),
        "errors": errors[:20],
    }


def split_by_game(
    samples: list[dict[str, Any]],
    valid_ratio: float = 0.2,
    seed: int = 20260706,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_game[str(sample["game_id"])].append(sample)

    game_ids = sorted(by_game)
    rng = random.Random(seed)
    rng.shuffle(game_ids)
    valid_games = set(game_ids[: max(1, round(len(game_ids) * valid_ratio))]) if len(game_ids) > 1 else set()

    train = [sample for gid in game_ids if gid not in valid_games for sample in by_game[gid]]
    valid = [sample for gid in game_ids if gid in valid_games for sample in by_game[gid]]
    if not train and valid:
        train, valid = valid, []

    split = {
        "split": "by_game_id",
        "seed": seed,
        "valid_ratio": valid_ratio,
        "train_games": sorted(set(game_ids) - valid_games),
        "valid_games": sorted(valid_games),
        "train_samples": len(train),
        "valid_samples": len(valid),
    }
    return train, valid, split


def schema_document(feature_names: list[str], tag_features: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "M6 V2 distillation samples. Inference features are derived only from visible observation fields.",
        "required_fields": sorted(REQUIRED_FIELDS),
        "feature_names": feature_names,
        "tag_features": tag_features,
        "split_rule": "train/valid must be split by game_id",
        "hidden_information_rule": "Hidden full state may be used by a teacher search, but must not be serialized into features/options.",
    }

