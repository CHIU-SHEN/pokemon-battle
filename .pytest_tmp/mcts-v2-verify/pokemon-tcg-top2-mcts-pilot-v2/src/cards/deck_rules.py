#!/usr/bin/env python3
"""Deck legality checks for Pokemon TCG AI Battle deck search."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARDS_JSON = PROJECT_ROOT / "data" / "cards.json"
TAGS_JSON = PROJECT_ROOT / "data" / "card_tags.json"


@dataclass
class DeckCheck:
    valid: bool
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_deck(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def card_tags(tags_db: dict[str, Any], card_id: int) -> set[str]:
    return set(tags_db["cards"].get(str(card_id), {}).get("tags", []))


def is_basic_energy(cards_db: dict[str, Any], tags_db: dict[str, Any], card_id: int) -> bool:
    card = cards_db["cards"].get(str(card_id))
    if not card:
        return False
    return "basic_energy" in card_tags(tags_db, card_id) or card.get("engine", {}).get("card_type") == 5


def is_basic_pokemon(cards_db: dict[str, Any], card_id: int) -> bool:
    card = cards_db["cards"].get(str(card_id))
    return bool(card and card.get("engine", {}).get("basic"))


def is_ace_spec(cards_db: dict[str, Any], tags_db: dict[str, Any], card_id: int) -> bool:
    card = cards_db["cards"].get(str(card_id))
    return bool(card and (card.get("engine", {}).get("ace_spec") or "ace_spec" in card_tags(tags_db, card_id)))


def check_deck(deck: list[int], cards_db: dict[str, Any] | None = None, tags_db: dict[str, Any] | None = None) -> DeckCheck:
    cards_db = cards_db or load_json(CARDS_JSON)
    tags_db = tags_db or load_json(TAGS_JSON)
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter(deck)

    if len(deck) != 60:
        errors.append(f"deck must contain exactly 60 cards, got {len(deck)}")
    invalid = sorted({cid for cid in deck if str(cid) not in cards_db["cards"]})
    if invalid:
        errors.append(f"invalid card IDs: {invalid[:10]}")

    for card_id, count in sorted(counts.items()):
        if str(card_id) not in cards_db["cards"]:
            continue
        if not is_basic_energy(cards_db, tags_db, card_id) and count > 4:
            errors.append(f"card {card_id} has {count} copies; non-basic-energy limit is 4")

    ace_count = sum(count for card_id, count in counts.items() if str(card_id) in cards_db["cards"] and is_ace_spec(cards_db, tags_db, card_id))
    if ace_count > 1:
        errors.append(f"ACE SPEC limit is 1, got {ace_count}")

    basic_pokemon_count = sum(count for card_id, count in counts.items() if str(card_id) in cards_db["cards"] and is_basic_pokemon(cards_db, card_id))
    if basic_pokemon_count < 1:
        errors.append("deck must include at least 1 Basic Pokemon")

    energy_count = sum(count for card_id, count in counts.items() if str(card_id) in cards_db["cards"] and is_basic_energy(cards_db, tags_db, card_id))
    if energy_count < 6:
        warnings.append(f"very low basic energy count: {energy_count}")

    return DeckCheck(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        counts={
            "total": len(deck),
            "unique": len(counts),
            "ace_spec": ace_count,
            "basic_pokemon": basic_pokemon_count,
            "basic_energy": energy_count,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck_csv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_deck(load_deck(Path(args.deck_csv)))
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print("valid" if result.valid else "invalid")
        for err in result.errors:
            print(f"ERROR: {err}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

