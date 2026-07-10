"""Visible-information feature extraction for M6 policy/value distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = PROJECT_ROOT / "submission"

import sys

if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from agent.deck_profile_abomasnow import (  # noqa: E402
    BASIC_WATER_ENERGY,
    KYOGRE,
    MAIN_ATTACKERS,
    MEGA_ABOMASNOW_EX,
    SNOVER,
)
from agent.parser import ParsedState, card_id_from_area, enum_value, safe_get, target_pokemon  # noqa: E402
from agent.rules import score_options  # noqa: E402
from cg.api import OptionType  # noqa: E402


TAG_FEATURES = [
    "basic_energy",
    "water_energy",
    "basic_pokemon",
    "evolution_pokemon",
    "ex",
    "main_attacker",
    "search_pokemon",
    "search_energy",
    "attach_energy",
    "draw",
    "supporter",
    "item",
    "tool",
    "damage_boost",
    "heal",
    "discard",
    "recursion",
    "deck_thinning",
    "ace_spec",
]

GLOBAL_FEATURES = [
    "turn",
    "turn_action_count",
    "supporter_played",
    "energy_attached",
    "own_prize_remaining",
    "opp_prize_remaining",
    "own_hand_count",
    "opp_hand_count",
    "own_deck_count",
    "opp_deck_count",
    "own_bench_count",
    "opp_bench_count",
    "own_active_hp",
    "own_active_max_hp",
    "own_active_damage",
    "own_active_energy",
    "opp_active_hp",
    "opp_active_max_hp",
    "opp_active_damage",
    "opp_active_energy",
    "own_discard_count",
    "own_discard_water_energy",
    "select_type",
    "select_context",
    "select_min",
    "select_max",
    "option_count",
]

OPTION_FEATURES = [
    "option_index",
    "option_type",
    "rule_score",
    "card_known",
    "card_id_scaled",
    "card_hash_00",
    "card_hash_01",
    "card_hash_02",
    "card_hash_03",
    "card_hash_04",
    "card_hash_05",
    "card_hash_06",
    "card_hash_07",
    "card_hash_08",
    "card_hash_09",
    "card_hash_10",
    "card_hash_11",
    "card_hash_12",
    "card_hash_13",
    "card_hash_14",
    "card_hash_15",
    "area",
    "player_index",
    "target_area",
    "target_player_index",
    "target_hp",
    "target_damage",
    "target_energy",
    "is_attack",
    "attack_id_scaled",
    "is_energy",
    "is_snover",
    "is_abomasnow",
    "is_kyogre",
] + [f"tag_{tag}" for tag in TAG_FEATURES]

FEATURE_NAMES = [f"global_{name}" for name in GLOBAL_FEATURES] + [f"option_{name}" for name in OPTION_FEATURES]


def _scale(value: float | int | None, denom: float, cap: float = 1.0) -> float:
    if value is None:
        return 0.0
    return max(-cap, min(cap, float(value) / denom))


def load_card_tags(path: Path | None = None) -> dict[int, set[str]]:
    tag_path = path or (PROJECT_ROOT / "data" / "card_tags.json")
    if not tag_path.exists():
        return {}
    raw = json.loads(tag_path.read_text(encoding="utf-8"))
    return {int(card_id): set(info.get("tags", [])) for card_id, info in raw.get("cards", {}).items()}


def _discard_water_count(parsed: ParsedState) -> int:
    return sum(1 for card in parsed.me.discard if safe_get(card, "id") == BASIC_WATER_ENERGY)


def global_features(parsed: ParsedState) -> list[float]:
    select = parsed.select
    me = parsed.me
    opp = parsed.opp
    own_active = me.active
    opp_active = opp.active
    return [
        _scale(parsed.turn, 50),
        _scale(parsed.turn_action_count, 20),
        float(parsed.supporter_played),
        float(parsed.energy_attached),
        _scale(me.prize_remaining, 6),
        _scale(opp.prize_remaining, 6),
        _scale(me.hand_count or len(me.hand), 20),
        _scale(opp.hand_count or len(opp.hand), 20),
        _scale(me.deck_count, 60),
        _scale(opp.deck_count, 60),
        _scale(len(me.bench), 5),
        _scale(len(opp.bench), 5),
        _scale(own_active.hp if own_active else None, 400),
        _scale(own_active.max_hp if own_active else None, 400),
        _scale(own_active.damage if own_active else None, 400),
        _scale(own_active.energy_count if own_active else None, 8),
        _scale(opp_active.hp if opp_active else None, 400),
        _scale(opp_active.max_hp if opp_active else None, 400),
        _scale(opp_active.damage if opp_active else None, 400),
        _scale(opp_active.energy_count if opp_active else None, 8),
        _scale(len(me.discard), 60),
        _scale(_discard_water_count(parsed), 20),
        _scale(select.type if select else None, 20),
        _scale(select.context if select else None, 60),
        _scale(select.min_count if select else None, 6),
        _scale(select.max_count if select else None, 6),
        _scale(len(select.options) if select else None, 20),
    ]


def option_card_id(parsed: ParsedState, option: Any) -> int | None:
    direct = safe_get(option, "cardId")
    if direct is not None:
        return int(direct)
    area = enum_value(safe_get(option, "area"))
    index = safe_get(option, "index")
    player_index = safe_get(option, "playerIndex", parsed.current_player)
    card_id = card_id_from_area(parsed, area, index, player_index)
    if card_id is not None:
        return int(card_id)
    in_play_area = enum_value(safe_get(option, "inPlayArea"))
    in_play_index = safe_get(option, "inPlayIndex")
    pokemon = target_pokemon(parsed, in_play_area, in_play_index, parsed.current_player)
    return int(pokemon.id) if pokemon is not None else None


def option_summary(parsed: ParsedState, option: Any, card_tags: dict[int, set[str]] | None = None) -> dict[str, Any]:
    card_id = option_card_id(parsed, option)
    tags = set(card_tags.get(card_id, set())) if card_tags and card_id is not None else set()
    area = enum_value(safe_get(option, "area"))
    player_index = safe_get(option, "playerIndex", parsed.current_player)
    target_area = enum_value(safe_get(option, "inPlayArea"))
    target_player_index = safe_get(option, "inPlayPlayerIndex", parsed.current_player)
    target = target_pokemon(parsed, target_area, safe_get(option, "inPlayIndex"), target_player_index)
    option_type = enum_value(safe_get(option, "type"))
    return {
        "option_type": option_type,
        "card_id": card_id,
        "tags": sorted(tags),
        "area": area,
        "player_index": player_index,
        "target_area": target_area,
        "target_player_index": target_player_index,
        "target_id": target.id if target else None,
        "attack_id": safe_get(option, "attackId"),
    }


def option_features(
    parsed: ParsedState,
    option: Any,
    option_index: int,
    rule_score: float,
    card_tags: dict[int, set[str]] | None = None,
) -> list[float]:
    card_id = option_card_id(parsed, option)
    tags = set(card_tags.get(card_id, set())) if card_tags and card_id is not None else set()
    option_type = enum_value(safe_get(option, "type"))
    area = enum_value(safe_get(option, "area"))
    player_index = safe_get(option, "playerIndex", parsed.current_player)
    target_area = enum_value(safe_get(option, "inPlayArea"))
    target_player_index = safe_get(option, "inPlayPlayerIndex", parsed.current_player)
    target = target_pokemon(parsed, target_area, safe_get(option, "inPlayIndex"), target_player_index)
    attack_id = safe_get(option, "attackId")

    hashed = [0.0] * 16
    if card_id is not None:
        hashed[int(card_id) % len(hashed)] = 1.0

    return [
        _scale(option_index, 20),
        _scale(option_type, 20),
        _scale(rule_score, 2000),
        float(card_id is not None),
        _scale(card_id, 2000),
        *hashed,
        _scale(area, 20),
        _scale(player_index, 2),
        _scale(target_area, 20),
        _scale(target_player_index, 2),
        _scale(target.hp if target else None, 400),
        _scale(target.damage if target else None, 400),
        _scale(target.energy_count if target else None, 8),
        float(option_type == enum_value(OptionType.ATTACK)),
        _scale(attack_id, 2000),
        float(card_id == BASIC_WATER_ENERGY or "basic_energy" in tags),
        float(card_id == SNOVER),
        float(card_id == MEGA_ABOMASNOW_EX or card_id in MAIN_ATTACKERS),
        float(card_id == KYOGRE),
        *[float(tag in tags) for tag in TAG_FEATURES],
    ]


def sample_features(parsed: ParsedState, card_tags: dict[int, set[str]] | None = None) -> tuple[list[float], list[list[float]], list[int], list[dict[str, Any]]]:
    select = parsed.select
    if select is None:
        return [], [], [], []
    tags = card_tags or {}
    global_vec = global_features(parsed)
    rule_scores = score_options(parsed)
    option_vecs = []
    summaries = []
    for idx, option in enumerate(select.options):
        local_vec = option_features(parsed, option, idx, rule_scores[idx] if idx < len(rule_scores) else 0.0, tags)
        option_vecs.append(global_vec + local_vec)
        summaries.append(option_summary(parsed, option, tags))
    return global_vec, option_vecs, rule_scores, summaries

