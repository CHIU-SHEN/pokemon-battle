#!/usr/bin/env python3
"""Build compact, leakage-audited Top10 Adapter sampling manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOP10 = ROOT / "data/high_score_decks"
DEFAULT_DATA = ROOT / "data/training/training_decisions_v1.jsonl"
DEFAULT_MANIFEST = ROOT / "data/training/training_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "data/adapter_views"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deck_cards(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path} contains {len(cards)} cards, expected 60")
    return cards


def sorted_deck_hash(cards: list[int]) -> str:
    canonical = "\n".join(str(card_id) for card_id in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def similarities(left: list[int], right: list[int]) -> dict[str, float]:
    a, b = Counter(left), Counter(right)
    keys = set(a) | set(b)
    dot = sum(a[key] * b[key] for key in keys)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    intersection = sum(min(a[key], b[key]) for key in keys)
    union = sum(max(a[key], b[key]) for key in keys)
    unique_intersection = len(set(a) & set(b))
    unique_union = len(set(a) | set(b))
    return {
        "multiset_cosine": cosine,
        "weighted_jaccard": intersection / union if union else 0.0,
        "unique_jaccard": unique_intersection / unique_union if unique_union else 0.0,
        "shared_cards": intersection,
    }


def classify(similarity: dict[str, float], *, exact: bool) -> str:
    if exact:
        return "exact"
    if similarity["multiset_cosine"] >= 0.80 and similarity["weighted_jaccard"] >= 0.50:
        return "similar"
    return "general"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_dataset(data_paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    decks: dict[str, dict[str, Any]] = {}
    game_splits: dict[str, set[str]] = defaultdict(set)
    unknown = Counter()
    total = 0
    source_rows = Counter()
    for data_path in data_paths:
      with data_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            source_rows[str(data_path.resolve())] += 1
            split = str(row["split"])
            game_id = str(row["game_id"])
            game_splits[game_id].add(split)
            player = (row.get("deck") or {}).get("player") or {}
            cards = player.get("cards")
            deck_hash = player.get("sha256_sorted_ids")
            if not cards or not deck_hash:
                unknown[split] += 1
                continue
            cards = [int(value) for value in cards]
            computed = sorted_deck_hash(cards)
            if computed.lower() != str(deck_hash).lower():
                raise ValueError(f"deck hash mismatch at {data_path}:{line_number}")
            key = computed.lower()
            entry = decks.setdefault(key, {
                "deck_sha256_sorted_ids": key,
                "cards": cards,
                "samples": Counter(),
                "games": defaultdict(set),
                "policy_sources": Counter(),
            })
            if sorted(entry["cards"]) != sorted(cards):
                raise ValueError(f"hash collision or inconsistent deck cards: {key}")
            entry["samples"][split] += 1
            entry["games"][split].add(game_id)
            source = (row.get("supervision") or {}).get("policy_source")
            if not source:
                source = "v0_rules" if (row.get("teacher") or {}).get("v0_action") is not None else "observed_action"
            entry["policy_sources"][str(source)] += 1
    cross_split = sorted(game for game, splits in game_splits.items() if len(splits) != 1)
    serializable = {}
    for key, entry in decks.items():
        serializable[key] = {
            "deck_sha256_sorted_ids": key,
            "cards": entry["cards"],
            "samples": dict(sorted(entry["samples"].items())),
            "games": {split: len(games) for split, games in sorted(entry["games"].items())},
            "policy_sources": dict(sorted(entry["policy_sources"].items())),
        }
    audit = {
        "rows": total,
        "source_rows": dict(source_rows),
        "known_deck_rows": total - sum(unknown.values()),
        "unknown_deck_rows": sum(unknown.values()),
        "unknown_by_split": dict(sorted(unknown.items())),
        "unique_known_decks": len(serializable),
        "unique_games": len(game_splits),
        "cross_split_games": cross_split,
    }
    return serializable, audit


def tier_totals(entries: list[dict[str, Any]]) -> dict[str, Any]:
    samples = Counter()
    games = Counter()
    for entry in entries:
        samples.update(entry["samples"])
        games.update(entry["games"])
    return {
        "deck_count": len(entries),
        "samples": dict(sorted(samples.items())),
        "games_sum_by_deck": dict(sorted(games.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--supplement", action="append", type=Path, default=[])
    args = parser.parse_args()
    training_manifest = load_json(args.training_manifest)
    candidates = []
    for metadata_path in sorted(TOP10.glob("*/metadata.json")):
        metadata = load_json(metadata_path)
        cards = deck_cards(metadata_path.parent / "deck.csv")
        deck_hash = sorted_deck_hash(cards)
        if deck_hash != str(metadata["deck_sha256_sorted_ids"]).lower():
            raise ValueError(f"candidate hash mismatch: {metadata['candidate_id']}")
        candidates.append({"metadata": metadata, "cards": cards, "hash": deck_hash})
    if len(candidates) != 10:
        raise ValueError(f"expected 10 candidates, found {len(candidates)}")

    data_paths = [args.data, *args.supplement]
    catalog, audit = catalog_dataset(data_paths)
    sources = [{
        "path": str(args.data.resolve()),
        "bytes": args.data.stat().st_size,
        "sha256": training_manifest["sha256"],
        "sha256_source": str(args.training_manifest.resolve()),
    }]
    sources.extend({
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    } for path in args.supplement)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "deck_catalog.json", {
        "schema_version": "adapter_deck_catalog_v1",
        "training_dataset_sha256": training_manifest["sha256"],
        "sources": sources,
        "audit": audit,
        "decks": catalog,
    })

    report_rows = []
    base_rows = audit["source_rows"].get(str(args.data.resolve()), 0)
    all_ok = not audit["cross_split_games"] and base_rows == int(training_manifest["samples"])
    for candidate in candidates:
        target_hash = candidate["hash"]
        tiers: dict[str, list[dict[str, Any]]] = {"exact": [], "similar": [], "general": []}
        ranked = []
        for deck_hash, entry in catalog.items():
            score = similarities(candidate["cards"], entry["cards"])
            tier = classify(score, exact=deck_hash == target_hash)
            item = {
                "deck_sha256_sorted_ids": deck_hash,
                **score,
                "samples": entry["samples"],
                "games": entry["games"],
            }
            tiers[tier].append(item)
            ranked.append({"tier": tier, **item})
        for entries in tiers.values():
            entries.sort(key=lambda item: (-item["multiset_cosine"], item["deck_sha256_sorted_ids"]))
        ranked.sort(key=lambda item: (-item["multiset_cosine"], item["deck_sha256_sorted_ids"]))
        exact_train = sum(item["samples"].get("train", 0) for item in tiers["exact"])
        exact_valid = sum(item["samples"].get("valid", 0) for item in tiers["exact"])
        view_ok = len(tiers["exact"]) == 1 and exact_train > 0
        all_ok = all_ok and view_ok
        view = {
            "schema_version": "adapter_sampling_view_v1",
            "candidate_id": candidate["metadata"]["candidate_id"],
            "archetype": candidate["metadata"]["archetype"],
            "variant": candidate["metadata"]["variant"],
            "target_deck_sha256_sorted_ids": target_hash,
            "training_dataset_sha256": training_manifest["sha256"],
            "sources": sources,
            "base_checkpoint": "artifacts/sl0_shared_full/best.pt",
            "classification": {
                "exact": "deck hash equals target hash",
                "similar": "multiset cosine >= 0.80 and weighted Jaccard >= 0.50",
                "general": "known player deck below similar thresholds",
                "unknown": "missing player deck; excluded from Adapter policy sampling",
            },
            "sampling": {
                "target_mix": {"exact": 0.40, "similar": 0.30, "general": 0.30},
                "replacement": {"exact": True, "similar": True, "general": False},
                "split_contract": "use the existing row split; never resplit samples or games",
                "unknown_policy": "exclude from Adapter policy; frozen shared trunk already learned general behavior",
            },
            "tiers": {
                "exact": tiers["exact"],
                "similar": tiers["similar"],
                "general_rule": "all catalog hashes not listed in exact or similar",
                "nearest_general": tiers["general"][:20],
            },
            "coverage": {
                "exact": tier_totals(tiers["exact"]),
                "similar": tier_totals(tiers["similar"]),
                "general": tier_totals(tiers["general"]),
                "unknown": {
                    "samples": audit["unknown_by_split"],
                    "excluded": True,
                },
            },
            "audit": {
                "exact_hash_found_once": len(tiers["exact"]) == 1,
                "exact_train_samples_positive": exact_train > 0,
                "exact_valid_samples_positive": exact_valid > 0,
                "cross_split_games": audit["cross_split_games"],
                "ok": view_ok and not audit["cross_split_games"],
            },
            "nearest_known_decks": ranked[:20],
        }
        candidate_dir = output / candidate["metadata"]["candidate_id"]
        write_json(candidate_dir / "view.json", view)
        report_rows.append({
            "candidate_id": view["candidate_id"],
            "exact_train": exact_train,
            "exact_valid": exact_valid,
            "exact_test": view["coverage"]["exact"]["samples"].get("test", 0),
            "similar_decks": view["coverage"]["similar"]["deck_count"],
            "similar_train": view["coverage"]["similar"]["samples"].get("train", 0),
            "ok": view["audit"]["ok"],
        })

    report = {
        "schema_version": "adapter_sampling_audit_v1",
        "training_dataset_sha256": training_manifest["sha256"],
        "sources": sources,
        "candidate_count": len(candidates),
        "dataset_audit": audit,
        "views": report_rows,
        "all_ok": all_ok,
    }
    write_json(output / "audit_report.json", report)
    lines = [
        "# Top10 Adapter 采样视图审计", "",
        f"> 数据集：`{training_manifest['sha256']}`  ",
        f"> 已知牌组样本：{audit['known_deck_rows']:,}；未知牌组样本：{audit['unknown_deck_rows']:,}  ",
        f"> 跨 split 对局：{len(audit['cross_split_games'])}；总体：{'通过' if all_ok else '未通过'}", "",
        "| Candidate | Exact train | Exact valid | Exact test | Similar decks | Similar train | Audit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['candidate_id']} | {row['exact_train']:,} | {row['exact_valid']:,} | "
            f"{row['exact_test']:,} | {row['similar_decks']:,} | {row['similar_train']:,} | "
            f"{'OK' if row['ok'] else 'FAIL'} |"
        )
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
