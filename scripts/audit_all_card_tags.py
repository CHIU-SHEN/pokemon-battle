"""Audit every local card and its generated/manual semantic tags."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cards.tags import infer_tags  # noqa: E402
from src.train.features import TAG_FEATURES  # noqa: E402


CARDS = ROOT / "data/cards.json"
TAGS = ROOT / "data/card_tags.json"
OVERRIDES = ROOT / "data/manual_overrides.json"
OUTPUT = ROOT / "data/card_tag_full_audit.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=CARDS)
    parser.add_argument("--tags", type=Path, default=TAGS)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    cards = load(args.cards)["cards"]
    tag_rows = load(args.tags)["cards"]
    overrides = load(args.overrides)
    allowed = {
        tag
        for row in tag_rows.values()
        for tag in row.get("tags", [])
    } | set(TAG_FEATURES) | {
        "attacker", "energy_discard", "gust", "hand_refresh",
        "mega_ex", "prize_acceleration", "risk_deckout", "setup_piece",
        "stadium_control", "water_energy",
    }

    rows = []
    all_errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for card_id in sorted(cards, key=int):
        card = cards[card_id]
        actual = set(tag_rows.get(card_id, {}).get("tags", []))
        automatic = infer_tags(card)
        override = overrides.get(card_id, {"add": [], "remove": []})
        added = set(override.get("add", []))
        removed = set(override.get("remove", []))
        expected = (automatic | added) - removed
        errors = []

        if not isinstance(card.get("card_id"), int) or card["card_id"] != int(card_id):
            errors.append("card_id mismatch")
        if not card.get("name_en") or not card.get("name_jp"):
            errors.append("missing bilingual card name")
        if not card.get("stage_type"):
            errors.append("missing stage_type")
        if actual != expected:
            errors.append(f"generated tags differ: expected={sorted(expected)} actual={sorted(actual)}")
        if added & removed:
            errors.append(f"same tags added and removed: {sorted(added & removed)}")
        unknown = (actual | added | removed) - allowed
        if unknown:
            errors.append(f"unknown tags: {sorted(unknown)}")
        if "basic_pokemon" in actual and "evolution_pokemon" in actual:
            errors.append("card cannot be both basic and evolution pokemon")
        if "basic_energy" in actual and "water_energy" in actual and card_id != "3":
            errors.append("water_energy assigned to a non-water basic energy id")

        stage = str(card.get("stage_type", "")).lower()
        expected_type_tags = {
            "supporter": "supporter" in stage,
            "item": "item" in stage,
            "tool": "tool" in stage,
            "basic_energy": "basic energy" in stage,
        }
        for tag, should_exist in expected_type_tags.items():
            if should_exist != (tag in actual):
                errors.append(f"type-derived tag mismatch: {tag}")

        row = {
            "card_id": int(card_id),
            "name_en": card.get("name_en"),
            "name_jp": card.get("name_jp"),
            "stage_type": card.get("stage_type"),
            "tags": sorted(actual),
            "automatic_tags": sorted(automatic),
            "manual_add": sorted(added),
            "manual_remove": sorted(removed),
            "review_method": "semantic_manual_override" if card_id in overrides else "full_rule_and_schema_audit",
            "errors": errors,
            "ok": not errors,
        }
        rows.append(row)
        counts["cards"] += 1
        counts["manual_override_cards"] += card_id in overrides
        counts["rule_audited_cards"] += card_id not in overrides
        counts["cards_with_errors"] += bool(errors)
        if errors:
            all_errors.append({"card_id": int(card_id), "name_en": card.get("name_en"), "errors": errors})

    orphan_overrides = sorted(set(overrides) - set(cards), key=int)
    if orphan_overrides:
        all_errors.append({"errors": [f"override ids absent from CardDB: {orphan_overrides}"]})

    report = {
        "schema_version": "card_tag_full_audit_v1",
        "scope": "all local CardDB records; semantic tags checked against local English effects and manual overrides",
        "source_limitation": "official source URLs, licensing, retrieval dates, and ruleset provenance are audited separately",
        "cards": counts["cards"],
        "manual_override_cards": counts["manual_override_cards"],
        "rule_audited_cards": counts["rule_audited_cards"],
        "cards_with_errors": counts["cards_with_errors"],
        "orphan_overrides": orphan_overrides,
        "model_tag_features": TAG_FEATURES,
        "errors": all_errors,
        "review_rows": rows,
        "ok": counts["cards"] == 1267 and not all_errors,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cards", "manual_override_cards", "rule_audited_cards", "cards_with_errors", "ok")}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
