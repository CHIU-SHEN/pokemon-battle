#!/usr/bin/env python3
"""Convert one observed-decision Adapter supplement to formal training rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_training_dataset import formal_row, iter_jsonl, supervision_for  # noqa: E402
from src.train.training_schema import validate_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "data/adapter_views/alakazam_battle_cage_split/exact_supplement_v1.jsonl",
    )
    parser.add_argument(
        "--view",
        type=Path,
        default=ROOT / "data/adapter_views/alakazam_battle_cage_split/view.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/adapter_views/alakazam_battle_cage_split/exact_supplement_training_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/adapter_views/alakazam_battle_cage_split/exact_supplement_training_v1.manifest.json",
    )
    parser.add_argument("--temperature", type=float, default=150.0)
    args = parser.parse_args()

    view = json.loads(args.view.read_text(encoding="utf-8"))
    target_hash = str(view["target_deck_sha256_sorted_ids"]).lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts = Counter()
    sample_ids: set[str] = set()
    games: dict[str, set[str]] = defaultdict(set)
    errors = []

    with args.output.open("wb") as output:
        for row in iter_jsonl(args.input):
            counts["input_rows"] += 1
            supervision = supervision_for(row, None, target_hash, args.temperature)
            sample = formal_row(row, supervision)
            sample_errors = validate_sample(sample)
            if sample_errors:
                counts["invalid_rows"] += 1
                if len(errors) < 100:
                    errors.append(
                        {"sample_id": row.get("sample_id"), "errors": sample_errors}
                    )
                continue
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                counts["duplicate_rows"] += 1
                if len(errors) < 100:
                    errors.append(
                        {"sample_id": sample_id, "errors": ["duplicate sample_id"]}
                    )
                continue
            sample_ids.add(sample_id)
            games[str(sample["game_id"])].add(str(sample["split"]))
            encoded = (
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            output.write(encoded)
            digest.update(encoded)
            counts["output_rows"] += 1
            counts[f"split:{sample['split']}"] += 1
            counts[f"policy:{supervision['policy_source']}"] += 1
            counts["empty_action_targets"] += int(
                supervision["empty_action_target"]
            )
            counts["forced_single_option"] += int(
                bool(sample["quality"].get("forced_single_option"))
            )
            counts["exact_target_deck"] += int(
                supervision["exact_target_deck"]
            )
            if supervision["target_action"]:
                legal = sample["legal_mask"]
                if any(
                    index < 0 or index >= len(legal) or not legal[index]
                    for index in supervision["target_action"]
                ):
                    counts["illegal_targets"] += 1

    cross_split_games = sorted(
        game_id for game_id, splits in games.items() if len(splits) > 1
    )
    ok = (
        counts["input_rows"] == counts["output_rows"]
        and counts["invalid_rows"] == 0
        and counts["duplicate_rows"] == 0
        and counts["illegal_targets"] == 0
        and not cross_split_games
        and counts["exact_target_deck"] == counts["output_rows"]
    )
    manifest = {
        "schema_version": "adapter_supplement_training_manifest_v1",
        "input": str(args.input),
        "input_schema": "observed_decision_v1",
        "output": str(args.output),
        "output_schema": "training_decision_v1",
        "sha256": digest.hexdigest().upper(),
        "bytes": args.output.stat().st_size,
        "target_deck_sha256_sorted_ids": target_hash,
        "samples": counts["output_rows"],
        "games": len(games),
        "splits": {
            split: counts[f"split:{split}"] for split in ("train", "valid", "test")
        },
        "policy_sources": {
            source: counts[f"policy:{source}"]
            for source in ("v0_rules", "observed_opponent")
        },
        "forced_single_option_samples": counts["forced_single_option"],
        "empty_action_targets": counts["empty_action_targets"],
        "exact_target_deck_samples": counts["exact_target_deck"],
        "illegal_targets": counts["illegal_targets"],
        "duplicate_rows": counts["duplicate_rows"],
        "invalid_rows": counts["invalid_rows"],
        "cross_split_games": cross_split_games,
        "errors": errors,
        "ok": ok,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
