#!/usr/bin/env python3
"""M5 deck legality, archetype generation, scoring, and MAP-Elites tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_CARDS = PROJECT_ROOT / "src" / "cards"
if str(SRC_CARDS) not in sys.path:
    sys.path.insert(0, str(SRC_CARDS))

from deck_rules import check_deck, load_deck, load_json  # noqa: E402


CARDS_JSON = PROJECT_ROOT / "data" / "cards.json"
TAGS_JSON = PROJECT_ROOT / "data" / "card_tags.json"
CANDIDATES_JSON = PROJECT_ROOT / "data" / "deck_candidates.json"
COEVOLUTION_JSON = PROJECT_ROOT / "data" / "deck_coevolution_plan.json"
ELITE_DIR = PROJECT_ROOT / "data" / "deck_elites"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    cards = load_json(CARDS_JSON)
    tags = load_json(TAGS_JSON)
    sample_deck = load_deck(PROJECT_ROOT / "submission" / "deck.csv")
    sample_check = check_deck(sample_deck, cards, tags)
    require(sample_check.valid, f"sample deck should be legal: {sample_check.errors}")

    short_check = check_deck(sample_deck[:-1], cards, tags)
    require(not short_check.valid and any("60" in e for e in short_check.errors), "short deck should be invalid")
    copy_check = check_deck([722] * 5 + [3] * 55, cards, tags)
    require(not copy_check.valid and any("limit is 4" in e for e in copy_check.errors), "5 Snover should be invalid")
    ace_check = check_deck([1158, 1158] + [722] * 4 + [723] * 4 + [3] * 50, cards, tags)
    require(not ace_check.valid and any("ACE SPEC" in e for e in ace_check.errors), "2 ACE SPEC should be invalid")

    candidates = json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))
    metadata = candidates["metadata"]
    require(metadata["generated_legal_variants"] >= 500, "expected at least 500 legal variants")
    for name, count in metadata["counts_by_archetype"].items():
        require(count >= 100, f"{name} generated only {count} variants")
    require(20 <= metadata["elite_count"] <= 50, f"elite count out of range: {metadata['elite_count']}")
    require(metadata["top10pct_proxy_mean"] > metadata["random_proxy_mean"] + 100, "top proxy decks should beat random proxy baseline")

    for elite in candidates["elites"]:
        require(elite["legal"]["valid"], f"illegal elite: {elite['archetype']}")
        require(len(elite["deck"]) == 60, "elite deck length mismatch")
        require("setup_stability" in elite["explanation"], "missing score explanation")

    exported = sorted(ELITE_DIR.glob("elite_*.csv"))
    require(len(exported) == metadata["elite_count"], "elite CSV export count mismatch")
    coevolution = json.loads(COEVOLUTION_JSON.read_text(encoding="utf-8"))
    require(coevolution["status"] == "proxy_screened_not_battle_promoted", "coevolution status should be conservative")
    require(coevolution["top_elite_summaries"], "missing top elite summaries")

    print("OK: deck optimizer, legality, MAP-Elites, and coevolution plan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

