"""Validate all materialized leaderboard Top10 deck candidates."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cards.deck_rules import check_deck, load_deck  # noqa: E402


def main() -> int:
    root = ROOT / "data/high_score_decks"
    manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    candidates = [path for path in root.iterdir() if path.is_dir()]
    assert manifest["candidate_count"] == len(candidates) == 10
    assert manifest["static_valid_count"] == 10
    assert manifest["promotion_decision"] != "promoted"
    for candidate in candidates:
        deck = load_deck(candidate / "deck.csv")
        report = check_deck(deck)
        mapping = json.loads((candidate / "mapping_report.json").read_text(encoding="utf-8"))
        assert len(deck) == 60
        assert report.valid, (candidate.name, report.errors)
        assert mapping["ok"] and not mapping["unmapped_card_ids"]
    print("OK: 10 leaderboard decks materialized and statically valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
