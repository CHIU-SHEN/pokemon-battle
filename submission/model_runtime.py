"""Pure-NumPy SL-0 inference used by the self-contained Kaggle package."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from agent.deck_profile_abomasnow import BASIC_WATER_ENERGY, KYOGRE, MAIN_ATTACKERS, MEGA_ABOMASNOW_EX, SNOVER
from agent.parser import ParsedState, card_id_from_area, enum_value, safe_get, target_pokemon
from agent.rules import score_options
from cg.api import OptionType


TAG_FEATURES = [
    "basic_energy", "water_energy", "basic_pokemon", "evolution_pokemon", "ex",
    "main_attacker", "search_pokemon", "search_energy", "attach_energy", "draw",
    "supporter", "item", "tool", "damage_boost", "heal", "discard", "recursion",
    "deck_thinning", "ace_spec",
]


def _scale(value, denom, cap=1.0):
    if value is None:
        return 0.0
    return max(-cap, min(cap, float(value) / denom))


def _candidate(name: str) -> str | None:
    for root in (os.getcwd(), "/kaggle_simulations/agent"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    return None


def load_tags() -> dict[int, set[str]]:
    path = _candidate("card_tags.json")
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return {int(card_id): set(info.get("tags", [])) for card_id, info in raw.get("cards", {}).items()}


def _discard_water_count(parsed: ParsedState) -> int:
    return sum(1 for card in parsed.me.discard if safe_get(card, "id") == BASIC_WATER_ENERGY)


def global_features(parsed: ParsedState) -> list[float]:
    select, me, opp = parsed.select, parsed.me, parsed.opp
    own_active, opp_active = me.active, opp.active
    return [
        _scale(parsed.turn, 50), _scale(parsed.turn_action_count, 20),
        float(parsed.supporter_played), float(parsed.energy_attached),
        _scale(me.prize_remaining, 6), _scale(opp.prize_remaining, 6),
        _scale(me.hand_count or len(me.hand), 20), _scale(opp.hand_count or len(opp.hand), 20),
        _scale(me.deck_count, 60), _scale(opp.deck_count, 60),
        _scale(len(me.bench), 5), _scale(len(opp.bench), 5),
        _scale(own_active.hp if own_active else None, 400), _scale(own_active.max_hp if own_active else None, 400),
        _scale(own_active.damage if own_active else None, 400), _scale(own_active.energy_count if own_active else None, 8),
        _scale(opp_active.hp if opp_active else None, 400), _scale(opp_active.max_hp if opp_active else None, 400),
        _scale(opp_active.damage if opp_active else None, 400), _scale(opp_active.energy_count if opp_active else None, 8),
        _scale(len(me.discard), 60), _scale(_discard_water_count(parsed), 20),
        _scale(select.type if select else None, 20), _scale(select.context if select else None, 60),
        _scale(select.min_count if select else None, 6), _scale(select.max_count if select else None, 6),
        _scale(len(select.options) if select else None, 20),
    ]


def option_card_id(parsed: ParsedState, option: Any) -> int | None:
    direct = safe_get(option, "cardId")
    if direct is not None:
        return int(direct)
    area, index = enum_value(safe_get(option, "area")), safe_get(option, "index")
    player_index = safe_get(option, "playerIndex", parsed.current_player)
    card_id = card_id_from_area(parsed, area, index, player_index)
    if card_id is not None:
        return int(card_id)
    target = target_pokemon(
        parsed, enum_value(safe_get(option, "inPlayArea")), safe_get(option, "inPlayIndex"), parsed.current_player
    )
    return int(target.id) if target is not None else None


def option_features(parsed: ParsedState, option: Any, index: int, rule_score: float,
                    tags_by_card: dict[int, set[str]]) -> list[float]:
    card_id = option_card_id(parsed, option)
    tags = tags_by_card.get(card_id, set()) if card_id is not None else set()
    option_type = enum_value(safe_get(option, "type"))
    area = enum_value(safe_get(option, "area"))
    player_index = safe_get(option, "playerIndex", parsed.current_player)
    target_area = enum_value(safe_get(option, "inPlayArea"))
    target_player = safe_get(option, "inPlayPlayerIndex", parsed.current_player)
    target = target_pokemon(parsed, target_area, safe_get(option, "inPlayIndex"), target_player)
    attack_id = safe_get(option, "attackId")
    hashed = [0.0] * 16
    if card_id is not None:
        hashed[int(card_id) % 16] = 1.0
    return [
        _scale(index, 20), _scale(option_type, 20), _scale(rule_score, 2000),
        float(card_id is not None), _scale(card_id, 2000), *hashed,
        _scale(area, 20), _scale(player_index, 2), _scale(target_area, 20), _scale(target_player, 2),
        _scale(target.hp if target else None, 400), _scale(target.damage if target else None, 400),
        _scale(target.energy_count if target else None, 8),
        float(option_type == enum_value(OptionType.ATTACK)), _scale(attack_id, 2000),
        float(card_id == BASIC_WATER_ENERGY or "basic_energy" in tags), float(card_id == SNOVER),
        float(card_id == MEGA_ABOMASNOW_EX or card_id in MAIN_ATTACKERS), float(card_id == KYOGRE),
        *[float(tag in tags) for tag in TAG_FEATURES],
    ]


def sample_features(parsed: ParsedState, tags: dict[int, set[str]]) -> tuple[np.ndarray, np.ndarray]:
    if parsed.select is None:
        raise ValueError("selection is required")
    global_vec = global_features(parsed)
    rule_scores = score_options(parsed)
    options = [
        global_vec + option_features(parsed, option, index, rule_scores[index], tags)
        for index, option in enumerate(parsed.select.options)
    ]
    return np.asarray(global_vec, dtype=np.float32), np.asarray(options, dtype=np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class SL0Policy:
    def __init__(self, deck: list[int]) -> None:
        model_path = _candidate("sl0_shared_best.npz")
        if model_path is None:
            raise FileNotFoundError("sl0_shared_best.npz")
        self.weights = np.load(model_path)
        self.deck = np.asarray(deck, dtype=np.int64)
        self.tags = load_tags()

    def _linear(self, prefix: str, x: np.ndarray) -> np.ndarray:
        return x @ self.weights[f"{prefix}.weight"].T + self.weights[f"{prefix}.bias"]

    def _layer_norm(self, prefix: str, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + 1e-5)
        return normalized * self.weights[f"{prefix}.weight"] + self.weights[f"{prefix}.bias"]

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        x = _gelu(self._layer_norm(f"{prefix}.1", self._linear(f"{prefix}.0", x)))
        return _gelu(self._layer_norm(f"{prefix}.5", self._linear(f"{prefix}.4", x)))

    def _deck_context(self) -> np.ndarray:
        embedding = self.weights["card_embedding.weight"]
        own = embedding[np.clip(self.deck, 0, embedding.shape[0] - 1)].mean(axis=0)
        opponent = np.zeros_like(own)
        return self._mlp("deck_encoder", np.concatenate([own, opponent]))

    def logits(self, parsed: ParsedState) -> np.ndarray:
        select = parsed.select
        if select is None or select.min_count != 1 or select.max_count != 1 or len(select.options) <= 1:
            raise ValueError("SL-0 is gated to non-trivial mandatory single-choice decisions")
        state_features, option_matrix = sample_features(parsed, self.tags)
        state = self._mlp("state_encoder", state_features)
        deck = self._deck_context()
        context = self._mlp("context", np.concatenate([state, deck]))
        options = self._mlp("option_encoder", option_matrix)
        logits = (options * context[None, :]).sum(axis=-1) / math.sqrt(float(context.shape[-1]))
        logits += self._linear("option_bias", options).reshape(-1)
        return logits

    def choose(self, parsed: ParsedState) -> list[int]:
        logits = self.logits(parsed)
        return [int(np.argmax(logits))]
