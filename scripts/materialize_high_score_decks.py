"""Materialize the ten leaderboard replay decks as validated engine-ID candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cards.deck_rules import check_deck  # noqa: E402


PROFILE_BY_CANDIDATE = {
    "alakazam_nighttime_mine": "high_score_02",
    "alakazam_neutralization_zone": "high_score_05",
    "alakazam_battle_cage_split": "high_score_10",
    "cynthia_garchomp_roserade": "high_score_08",
    "crustle_kangaskhan_cage": "high_score_09",
    "crustle_kangaskhan_petrel": "high_score_03",
    "marnie_grimmsnarl_froslass": "high_score_07",
    "marnie_grimmsnarl_dudunsparce": "high_score_04",
    "marnie_grimmsnarl_tatsugiri": "high_score_01",
    "mega_starmie_dusknoir": "high_score_06",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deck_ids(deck: dict[str, int]) -> list[int]:
    return sorted(int(card_id) for card_id, count in deck.items() for _ in range(int(count)))


def deck_hash(ids: list[int]) -> str:
    return hashlib.sha256("\n".join(map(str, sorted(ids))).encode("utf-8")).hexdigest()


def wilson_lower(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    rate = wins / games
    denominator = 1.0 + z * z / games
    centre = rate + z * z / (2.0 * games)
    adjustment = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * games)) / games)
    return (centre - adjustment) / denominator


def main() -> int:
    selection = load(ROOT / "data/high_score_deck_selection.json")
    core = load(ROOT / "data/external/kaggle_replays/core_combo_candidates.json")
    cards = load(ROOT / "data/cards.json")
    tags = load(ROOT / "data/card_tags.json")
    profiles = {item["profile_id"]: item for item in core["profiles"]}
    candidates = {item["candidate_id"]: item for item in selection["candidates"]}
    output_root = ROOT / "data/high_score_decks"
    output_root.mkdir(parents=True, exist_ok=True)
    ranking = []
    candidate_hashes = {
        candidate_id: deck_hash(deck_ids(profiles[profile_id]["deck"]))
        for candidate_id, profile_id in PROFILE_BY_CANDIDATE.items()
    }
    candidate_by_hash = {value: key for key, value in candidate_hashes.items()}
    coverage_counts: dict[str, Counter] = {candidate_id: Counter() for candidate_id in candidate_hashes}
    coverage_games: dict[str, set[str]] = {candidate_id: set() for candidate_id in candidate_hashes}
    converted_path = ROOT / "data/processed/kaggle_decisions.jsonl"
    if converted_path.exists():
        with converted_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                player_hash = (((row.get("deck") or {}).get("player") or {}).get("sha256_sorted_ids"))
                candidate_id = candidate_by_hash.get(player_hash)
                if candidate_id is None:
                    continue
                coverage_counts[candidate_id]["samples"] += 1
                coverage_counts[candidate_id][f"split:{row['split']}"] += 1
                coverage_games[candidate_id].add(row["game_id"])

    for candidate_id, profile_id in PROFILE_BY_CANDIDATE.items():
        candidate = candidates[candidate_id]
        profile = profiles[profile_id]
        ids = deck_ids(profile["deck"])
        validation = check_deck(ids, cards, tags)
        exact = [
            item for item in core["matches"]
            if item["target_profile"] == profile_id and float(item["deck_similarity"]) == 1.0
        ]
        all_matches = [item for item in core["matches"] if item["target_profile"] == profile_id]
        exact_wins = sum(bool(item["won"]) for item in exact)
        all_wins = sum(bool(item["won"]) for item in all_matches)
        candidate_dir = output_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "deck.csv").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")

        metadata = {
            "schema_version": "high_score_deck_candidate_v1",
            "candidate_id": candidate_id,
            "archetype": candidate["archetype"],
            "variant": candidate["variant"],
            "source_profile": profile_id,
            "source_replay_ids": sorted(set(int(x) for x in candidate["replays"])),
            "profile_seed_episodes": profile["seed_episodes"],
            "profile_seed_teams": profile["seed_teams"],
            "deck_sha256_sorted_ids": deck_hash(ids),
            "card_count": len(ids),
            "unique_card_count": len(set(ids)),
            "promotion_status": "static_validated_awaiting_strategy_screen",
        }
        mapping_rows = []
        for card_id, count in sorted(Counter(ids).items()):
            card = cards["cards"].get(str(card_id))
            mapping_rows.append({
                "card_id": card_id,
                "count": count,
                "name_en": None if card is None else card.get("name_en"),
                "name_jp": None if card is None else card.get("name_jp"),
                "stage_type": None if card is None else card.get("stage_type"),
                "mapping_source": "exact engine card ID reconstructed from complete replay deck",
                "mapped": card is not None,
            })
        mapping_report = {
            "schema_version": "high_score_deck_mapping_v1",
            "candidate_id": candidate_id,
            "source_profile": profile_id,
            "method": "complete replay opening deck action; no name-only inference",
            "cards": mapping_rows,
            "unmapped_card_ids": [row["card_id"] for row in mapping_rows if not row["mapped"]],
            "ok": all(row["mapped"] for row in mapping_rows),
        }
        static_report = {
            "schema_version": "high_score_deck_static_validation_v1",
            "candidate_id": candidate_id,
            **asdict(validation),
        }
        evidence = {
            "schema_version": "high_score_deck_replay_prior_v1",
            "candidate_id": candidate_id,
            "source_profile": profile_id,
            "all_similarity_matches": len(all_matches),
            "all_similarity_wins": all_wins,
            "all_similarity_win_rate": round(all_wins / len(all_matches), 6) if all_matches else None,
            "exact_deck_trajectories": len(exact),
            "exact_deck_wins": exact_wins,
            "exact_deck_win_rate": round(exact_wins / len(exact), 6) if exact else None,
            "exact_deck_wilson95_lower": round(wilson_lower(exact_wins, len(exact)), 6),
            "converted_training_coverage": {
                "decision_samples": coverage_counts[candidate_id]["samples"],
                "games": len(coverage_games[candidate_id]),
                "splits": {
                    name: coverage_counts[candidate_id][f"split:{name}"]
                    for name in ("train", "valid", "test")
                },
            },
            "selection_use": "prior only; not a fair head-to-head promotion result",
        }
        for name, doc in (
            ("metadata.json", metadata),
            ("mapping_report.json", mapping_report),
            ("static_validation.json", static_report),
            ("replay_prior.json", evidence),
        ):
            (candidate_dir / name).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        ranking.append({**metadata, **evidence, "static_valid": validation.valid})

    ranking.sort(key=lambda item: item["exact_deck_wilson95_lower"], reverse=True)
    manifest = {
        "schema_version": "high_score_deck_materialization_v1",
        "candidate_count": len(ranking),
        "static_valid_count": sum(bool(item["static_valid"]) for item in ranking),
        "mapping_complete_count": len(ranking),
        "incumbent_submission": {
            "role": "legacy baseline pending Top10 tournament",
            "profile_id": "current_submission",
        },
        "replay_prior_leader": ranking[0]["candidate_id"] if ranking else None,
        "promotion_decision": "pending common-policy screen and deck-adapted finals",
        "ranking_by_exact_replay_wilson_lower_bound": ranking,
        "ok": len(ranking) == 10 and all(item["static_valid"] for item in ranking),
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": manifest["candidate_count"],
        "static_valid_count": manifest["static_valid_count"],
        "replay_prior_leader": manifest["replay_prior_leader"],
        "ok": manifest["ok"],
    }, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
