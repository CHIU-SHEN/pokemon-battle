#!/usr/bin/env python3
"""M4 CardDB and tag artifact checks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARDS = PROJECT_ROOT / "data" / "cards.json"
TAGS = PROJECT_ROOT / "data" / "card_tags.json"
OVERRIDES = PROJECT_ROOT / "data" / "manual_overrides.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    cards_db = json.loads(CARDS.read_text(encoding="utf-8"))
    tags_db = json.loads(TAGS.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))

    require(cards_db["metadata"]["unique_card_ids"] == 1267, "CardDB unique count mismatch")
    require(cards_db["metadata"]["engine_card_count"] == 1267, "engine card count mismatch")
    require(cards_db["metadata"]["ruleset"]["id"] == "ptcg_abc_2026_simulation_designated_pool_v1", "ruleset mismatch")
    require(cards_db["metadata"]["ruleset"]["designated_card_count"] == 1267, "ruleset card pool mismatch")
    require(cards_db["metadata"]["ruleset"]["player_time_limit_seconds"] == 600, "ruleset clock mismatch")
    require(cards_db["metadata"]["row_counts"]["en_csv"] == 2022, "EN row count mismatch")
    require(cards_db["metadata"]["row_counts"]["jp_csv"] == 2022, "JP row count mismatch")
    require(len(cards_db["cards"]) == len(tags_db["cards"]) == 1267, "card/tag count mismatch")

    aboma = cards_db["cards"]["723"]
    require(aboma["name_en"] == "Mega Abomasnow ex", "wrong Abomasnow name")
    require(aboma["engine"]["stage1"] is True, "Abomasnow should be stage1")
    require(aboma["engine"]["mega_ex"] is True, "Abomasnow should be mega ex")
    require(any(a["name_en"] == "Frost Barrier" for a in aboma["attacks"]), "missing Frost Barrier")

    expected_tags = {
        "3": {"basic_energy", "water_energy"},
        "722": {"basic_pokemon", "setup_piece"},
        "723": {"main_attacker", "mega_ex", "risk_deckout"},
        "1145": {"search_pokemon", "deck_thinning"},
        "1158": {"ace_spec", "damage_boost", "tool"},
        "1205": {"supporter", "search_pokemon"},
        "1227": {"supporter", "draw", "hand_refresh"},
        "1235": {"supporter", "search_energy", "attach_energy"},
    }
    for card_id, required in expected_tags.items():
        actual = set(tags_db["cards"][card_id]["tags"])
        missing = required - actual
        require(not missing, f"{card_id} missing tags: {sorted(missing)}")
    require(len(overrides) >= 9, "manual overrides should cover main deck cards")
    print("OK: CardDB 1267 cards and key tags passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

