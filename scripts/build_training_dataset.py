"""Merge observed, V0, and V1 data into the first formal training dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.training_schema import SCHEMA_VERSION, validate_sample  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def hard_policy(action: list[int], option_count: int) -> list[float]:
    policy = [0.0] * option_count
    if action:
        mass = 1.0 / len(action)
        for index in action:
            policy[index] += mass
    return policy


def v1_policy(label: dict[str, Any], option_count: int, temperature: float) -> list[float]:
    scores = (label.get("search") or {}).get("scores") or []
    valid = [item for item in scores if isinstance(item.get("action"), list) and item.get("score") is not None]
    if not valid:
        return hard_policy(label["v1_action"], option_count)
    maximum = max(float(item["score"]) for item in valid)
    weights = [math.exp((float(item["score"]) - maximum) / temperature) for item in valid]
    total = sum(weights)
    policy = [0.0] * option_count
    for item, weight in zip(valid, weights):
        action = [int(i) for i in item["action"]]
        if not action:
            continue
        mass = weight / total / len(action)
        for index in action:
            if 0 <= index < option_count:
                policy[index] += mass
    policy_total = sum(policy)
    if policy_total <= 0:
        return hard_policy(label["v1_action"], option_count)
    return [value / policy_total for value in policy]


def supervision_for(
    row: dict[str, Any],
    v1: dict[str, Any] | None,
    target_hash: str,
    temperature: float,
) -> dict[str, Any]:
    option_count = int(row["select"]["option_count"])
    v0 = (row.get("teacher") or {}).get("v0_action")
    source_type = (row.get("source") or {}).get("type")
    player_hash = (((row.get("deck") or {}).get("player") or {}).get("sha256_sorted_ids"))
    exact_target_deck = bool(player_hash and player_hash == target_hash)

    if v1 is not None:
        policy_source = "v1_search"
        target_action = [int(i) for i in v1["v1_action"]]
        # The option vector has no dedicated "stop" slot. When search chooses
        # the legal empty action, represent it with an all-zero option policy.
        soft_policy = (
            v1_policy(v1, option_count, temperature)
            if target_action
            else [0.0] * option_count
        )
        base_weight = 2.0
        confidence = max(soft_policy, default=0.0)
    elif v0 is not None:
        policy_source = "v0_rules"
        target_action = [int(i) for i in v0]
        soft_policy = hard_policy(target_action, option_count)
        base_weight = 1.4
        confidence = 0.8
    elif source_type == "kaggle_official_replay":
        policy_source = "kaggle_agent"
        target_action = [int(i) for i in row["observed_action"]]
        soft_policy = hard_policy(target_action, option_count)
        base_weight = 1.25 if exact_target_deck else 0.7
        confidence = 0.65
    else:
        policy_source = "observed_opponent"
        target_action = [int(i) for i in row["observed_action"]]
        soft_policy = hard_policy(target_action, option_count)
        base_weight = 0.6
        confidence = 0.5

    forced = bool((row.get("quality") or {}).get("forced_single_option"))
    sample_weight = base_weight * (0.15 if forced else 1.0)
    value_available = row.get("value_target") is not None
    return {
        "policy_source": policy_source,
        "target_action": target_action,
        "empty_action_target": not target_action,
        "soft_policy": soft_policy,
        "confidence": confidence,
        "sample_weight": sample_weight,
        "head_weights": {
            "policy": sample_weight,
            "value": sample_weight * 0.5 if value_available else 0.0,
            "risk": 0.0,
        },
        "v0_action": v0,
        "v1": None if v1 is None else {
            "action": v1["v1_action"],
            "changed_v0": v1["v1_changed_v0"],
            "search": v1["search"],
            "budget": v1["budget"],
        },
        "exact_target_deck": exact_target_deck,
    }


def formal_row(row: dict[str, Any], supervision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "game_id": row["game_id"],
        "split": row["split"],
        "source": row["source"],
        "turn": row["turn"],
        "step": row["step"],
        "current_player": row["current_player"],
        "select": row["select"],
        "legal_mask": row["legal_mask"],
        "features": row["features"],
        "options": row["options"],
        "option_features": row["option_features"],
        "public_history": row["public_history"],
        "deck": row["deck"],
        "observed_action": row["observed_action"],
        "game_result": row["game_result"],
        "value_target": row["value_target"],
        "supervision": supervision,
        "quality": row["quality"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-cases", type=Path, default=ROOT / "data/processed/bad_case_decisions.jsonl")
    parser.add_argument("--kaggle", type=Path, default=ROOT / "data/processed/kaggle_decisions.jsonl")
    parser.add_argument("--v1", type=Path, default=ROOT / "data/reanalysis/v1_labels.jsonl")
    parser.add_argument("--target-profile", type=Path, default=ROOT / "data/processed/target_deck_profile.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/training_manifest_v1.json")
    parser.add_argument("--temperature", type=float, default=150.0)
    args = parser.parse_args()

    target_profile = load_json(args.target_profile)
    target_hash = str(target_profile["sha256_sorted_ids"])
    v1_labels = {row["sample_id"]: row for row in iter_jsonl(args.v1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    sample_ids: set[str] = set()
    games: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    errors = []

    with args.output.open("wb") as output:
        for input_path in (args.bad_cases, args.kaggle):
            for row in iter_jsonl(input_path):
                supervision = supervision_for(row, v1_labels.get(row["sample_id"]), target_hash, args.temperature)
                sample = formal_row(row, supervision)
                sample_errors = validate_sample(sample)
                if sample_errors:
                    if len(errors) < 100:
                        errors.append({"sample_id": row.get("sample_id"), "errors": sample_errors})
                    continue
                encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(encoded)
                digest.update(encoded)
                sample_id = sample["sample_id"]
                if sample_id in sample_ids:
                    errors.append({"sample_id": sample_id, "errors": ["duplicate sample_id"]})
                sample_ids.add(sample_id)
                games[sample["game_id"]].add(sample["split"])
                counts["samples"] += 1
                counts[f"split:{sample['split']}"] += 1
                counts[f"policy:{supervision['policy_source']}"] += 1
                counts["forced_single_option"] += bool(sample["quality"].get("forced_single_option"))
                counts["empty_action_target"] += supervision["empty_action_target"]
                counts["exact_target_deck"] += supervision["exact_target_deck"]

    cross_split_games = sorted(game_id for game_id, splits in games.items() if len(splits) > 1)
    unused_v1 = sorted(set(v1_labels) - sample_ids)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_deck_sha256": target_hash,
        "target_deck_role": target_profile.get("role", "unspecified"),
        "output": display_path(args.output),
        "sha256": digest.hexdigest().upper(),
        "bytes": args.output.stat().st_size,
        "samples": counts["samples"],
        "games": len(games),
        "splits": {name: counts[f"split:{name}"] for name in ("train", "valid", "test")},
        "policy_sources": {
            name: counts[f"policy:{name}"]
            for name in ("v1_search", "v0_rules", "kaggle_agent", "observed_opponent")
        },
        "forced_single_option_samples": counts["forced_single_option"],
        "empty_action_targets": counts["empty_action_target"],
        "exact_target_deck_samples": counts["exact_target_deck"],
        "v1_labels_loaded": len(v1_labels),
        "unused_v1_labels": len(unused_v1),
        "unique_sample_ids": len(sample_ids),
        "cross_split_games": cross_split_games,
        "temperature": args.temperature,
        "weighting": {
            "v1_search": 2.0,
            "v0_rules": 1.4,
            "kaggle_exact_target": 1.25,
            "kaggle_other": 0.7,
            "observed_opponent": 0.6,
            "forced_single_option_multiplier": 0.15,
            "value_head_multiplier": 0.5,
            "risk_head": "disabled until a reliable target exists",
        },
        "errors": errors,
        "ok": not errors and not cross_split_games and not unused_v1 and counts["samples"] > 0,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
