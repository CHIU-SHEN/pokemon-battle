#!/usr/bin/env python3
"""Automatic and manual card effect tags for CardDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARDS_JSON = PROJECT_ROOT / "data" / "cards.json"
TAGS_JSON = PROJECT_ROOT / "data" / "card_tags.json"
OVERRIDES_JSON = PROJECT_ROOT / "data" / "manual_overrides.json"


TAG_PATTERNS: dict[str, list[str]] = {
    "draw": [r"\bdraw\b", r"put .* into your hand"],
    "search_pokemon": [r"search your deck .*pok[eé]mon", r"pok[eé]mon .* into your hand"],
    "search_energy": [r"search your deck .*energy", r"basic energy card you find"],
    "attach_energy": [r"attach .*energy", r"attach a basic energy"],
    "switch": [r"\bswitch\b", r"switch your active"],
    "gust": [r"opponent.*bench.*active", r"switch .* opponent"],
    "discard": [r"\bdiscard\b"],
    "damage_boost": [r"do \d+ more damage", r"does \d+ more damage"],
    "heal": [r"\bheal\b", r"recover .*hp"],
    "stadium_control": [r"\bstadium\b"],
    "deck_thinning": [r"search your deck", r"look at the top"],
    "recursion": [r"from your discard pile", r"put .* discard .* into your hand", r"shuffle .* into your deck"],
    "prize_acceleration": [r"prize card", r"take .* prize"],
    "risk_deckout": [r"discard the top", r"mill"],
}


DEFAULT_MANUAL_OVERRIDES = {
    "3": {"add": ["basic_energy", "water_energy"], "remove": []},
    "721": {"add": ["attacker", "energy_discard", "recursion"], "remove": []},
    "722": {"add": ["basic_pokemon", "setup_piece"], "remove": []},
    "723": {"add": ["main_attacker", "mega_ex", "risk_deckout"], "remove": []},
    "1145": {"add": ["search_pokemon", "deck_thinning"], "remove": []},
    "1158": {"add": ["ace_spec", "damage_boost", "tool"], "remove": []},
    "1205": {"add": ["supporter", "search_pokemon", "deck_thinning"], "remove": []},
    "1227": {"add": ["supporter", "draw", "hand_refresh"], "remove": []},
    "1235": {"add": ["supporter", "search_energy", "attach_energy", "deck_thinning"], "remove": []},
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_manual_overrides(path: Path = OVERRIDES_JSON) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    write_json(path, DEFAULT_MANUAL_OVERRIDES)
    return DEFAULT_MANUAL_OVERRIDES


def text_blob(card: dict[str, Any]) -> str:
    chunks = [
        card.get("name_en", ""),
        card.get("stage_type", ""),
        card.get("rule_box", ""),
        card.get("category", ""),
        " ".join(skill.get("text", "") for skill in card.get("engine", {}).get("skills", [])),
    ]
    for attack in card.get("attacks", []):
        chunks.extend([attack.get("name_en", ""), attack.get("text_en", "")])
    return " ".join(chunks).lower()


def infer_tags(card: dict[str, Any]) -> set[str]:
    blob = text_blob(card)
    tags: set[str] = set()
    engine = card.get("engine", {})
    stage_type = card.get("stage_type", "").lower()
    if "supporter" in stage_type:
        tags.add("supporter")
    if "item" in stage_type:
        tags.add("item")
    if "tool" in stage_type:
        tags.add("tool")
    if "stadium" in stage_type:
        tags.add("stadium_control")
    if "basic energy" in stage_type:
        tags.add("basic_energy")
    if engine.get("ace_spec"):
        tags.add("ace_spec")
    if engine.get("ex"):
        tags.add("ex")
    if engine.get("mega_ex"):
        tags.add("mega_ex")
    if engine.get("basic"):
        tags.add("basic_pokemon")
    if engine.get("stage1") or engine.get("stage2"):
        tags.add("evolution_pokemon")
    for tag, patterns in TAG_PATTERNS.items():
        if any(re.search(pattern, blob) for pattern in patterns):
            tags.add(tag)
    return tags


def build_tags(cards_json: Path = CARDS_JSON, overrides_json: Path = OVERRIDES_JSON) -> dict[str, Any]:
    db = load_json(cards_json)
    overrides = ensure_manual_overrides(overrides_json)
    tags: dict[str, Any] = {}
    for card_id, card in db["cards"].items():
        inferred = infer_tags(card)
        override = overrides.get(card_id, {})
        inferred.update(override.get("add", []))
        inferred.difference_update(override.get("remove", []))
        tags[card_id] = {
            "name_en": card.get("name_en"),
            "tags": sorted(inferred),
            "source": "auto+manual" if card_id in overrides else "auto",
        }
    metadata = {
        "card_count": len(tags),
        "tag_count": len({tag for item in tags.values() for tag in item["tags"]}),
        "manual_override_count": len(overrides),
    }
    return {"metadata": metadata, "cards": tags}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", default=str(CARDS_JSON))
    parser.add_argument("--overrides", default=str(OVERRIDES_JSON))
    parser.add_argument("--out", default=str(TAGS_JSON))
    args = parser.parse_args()
    tags = build_tags(Path(args.cards), Path(args.overrides))
    write_json(Path(args.out), tags)
    print(json.dumps(tags["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

