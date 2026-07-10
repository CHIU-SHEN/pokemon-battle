"""Constrained hidden-state particles for Search API calls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import random
from typing import Any

from cg.api import Observation, to_observation_class

from .deck_profile_abomasnow import BASIC_WATER_ENERGY, SNOVER
from .parser import GameLedger, safe_get


DEFAULT_DECK = [
    1158,
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1145,
    1145,
    1145,
    1145,
    1205,
    1205,
    1227,
    1227,
    1227,
    1227,
    1235,
    1235,
    1235,
    1235,
] + [BASIC_WATER_ENERGY] * 35


@dataclass
class BeliefParticle:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def read_deck(deck_path: str | None = None) -> list[int]:
    candidates = []
    if deck_path:
        candidates.append(deck_path)
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "deck.csv"))
    candidates.append("/kaggle_simulations/agent/deck.csv")
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    return list(DEFAULT_DECK)


def _card_id(card: Any) -> int | None:
    return safe_get(card, "id") if card is not None else None


def _pokemon_public_ids(pokemon: Any) -> list[int]:
    if pokemon is None:
        return []
    ids: list[int] = []
    card_id = safe_get(pokemon, "id")
    if card_id is not None:
        ids.append(int(card_id))
    for attr in ("preEvolution", "energyCards", "tools"):
        for card in safe_get(pokemon, attr, []) or []:
            cid = _card_id(card)
            if cid is not None:
                ids.append(int(cid))
    return ids


def _visible_ids_for_player(player: Any) -> list[int]:
    ids: list[int] = []
    for card in safe_get(player, "hand", []) or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(int(cid))
    for card in safe_get(player, "discard", []) or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(int(cid))
    for card in safe_get(player, "prize", []) or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(int(cid))
    for pokemon in safe_get(player, "active", []) or []:
        ids.extend(_pokemon_public_ids(pokemon))
    for pokemon in safe_get(player, "bench", []) or []:
        ids.extend(_pokemon_public_ids(pokemon))
    return ids


def _remaining_pool(deck: list[int], visible_ids: list[int]) -> list[int]:
    counts = Counter(deck)
    for cid in visible_ids:
        if counts[cid] > 0:
            counts[cid] -= 1
    pool: list[int] = []
    for cid, count in counts.items():
        pool.extend([cid] * max(0, count))
    return pool


def _draw_from_pool(pool: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    rng.shuffle(pool)
    if len(pool) < count:
        pool.extend([BASIC_WATER_ENERGY] * (count - len(pool)))
    drawn = pool[:count]
    del pool[:count]
    return drawn


class BeliefSampler:
    """Samples legal-size hidden states from public observations."""

    def __init__(self, deck: list[int] | None = None, seed: int = 20260706) -> None:
        self.deck = list(deck or read_deck())
        self.rng = random.Random(seed)

    def _sample_player_hidden(self, player: Any, known_deck: list[int]) -> tuple[list[int], list[int], list[int]]:
        pool = _remaining_pool(known_deck, _visible_ids_for_player(player))
        prize = safe_get(player, "prize", []) or []
        hidden_prize_count = sum(1 for card in prize if card is None)
        deck_count = int(safe_get(player, "deckCount", 0) or 0)
        hand_count = int(safe_get(player, "handCount", 0) or 0)
        known_hand = safe_get(player, "hand", None)
        hidden_hand_count = hand_count if known_hand is None else 0
        hidden_prize = _draw_from_pool(pool, hidden_prize_count, self.rng)
        hidden_deck = _draw_from_pool(pool, deck_count, self.rng)
        hidden_hand = _draw_from_pool(pool, hidden_hand_count, self.rng)
        return hidden_deck, hidden_prize, hidden_hand

    def sample(self, obs_or_dict: dict | Observation, ledger: GameLedger | None = None) -> BeliefParticle:
        obs: Observation = obs_or_dict if isinstance(obs_or_dict, Observation) else to_observation_class(obs_or_dict)
        state = obs.current
        if state is None:
            raise ValueError("cannot sample belief before battle start")
        your_index = state.yourIndex
        opp_index = 1 - your_index
        your_player = state.players[your_index]
        opp_player = state.players[opp_index]
        your_deck, your_prize, _ = self._sample_player_hidden(your_player, self.deck)
        opp_deck, opp_prize, opp_hand = self._sample_player_hidden(opp_player, self.deck)

        opponent_active: list[int] = []
        active = safe_get(opp_player, "active", []) or []
        if active and active[0] is None:
            opponent_active = [SNOVER]

        return BeliefParticle(
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=opp_deck,
            opponent_prize=opp_prize,
            opponent_hand=opp_hand,
            opponent_active=opponent_active,
        )

    def validate(self, obs_or_dict: dict | Observation, particle: BeliefParticle) -> bool:
        obs: Observation = obs_or_dict if isinstance(obs_or_dict, Observation) else to_observation_class(obs_or_dict)
        state = obs.current
        if state is None:
            return False
        your_index = state.yourIndex
        opp_index = 1 - your_index
        your_player = state.players[your_index]
        opp_player = state.players[opp_index]
        active = safe_get(opp_player, "active", []) or []
        need_active = bool(active and active[0] is None)
        return (
            len(particle.your_deck) >= int(safe_get(your_player, "deckCount", 0) or 0)
            and len(particle.your_prize) >= sum(1 for c in (safe_get(your_player, "prize", []) or []) if c is None)
            and len(particle.opponent_deck) >= int(safe_get(opp_player, "deckCount", 0) or 0)
            and len(particle.opponent_prize) >= sum(1 for c in (safe_get(opp_player, "prize", []) or []) if c is None)
            and len(particle.opponent_hand) >= int(safe_get(opp_player, "handCount", 0) or 0)
            and (not need_active or bool(particle.opponent_active))
        )

