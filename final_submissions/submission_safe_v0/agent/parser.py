"""Observation parsing helpers for the V0 agent.

The official dataclasses may grow during the competition, so this module keeps
the internal view intentionally small and tolerant of missing attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cg.api import AreaType, Observation, to_observation_class


def enum_value(value: Any) -> int | None:
    if value is None:
        return None
    return getattr(value, "value", value)


def safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class ParsedPokemon:
    id: int
    serial: int
    hp: int
    max_hp: int
    area: int
    index: int
    player_index: int
    energy_count: int = 0
    tool_count: int = 0
    appear_this_turn: bool = False

    @property
    def damage(self) -> int:
        return max(0, self.max_hp - self.hp)


@dataclass
class ParsedPlayer:
    index: int
    active: ParsedPokemon | None = None
    bench: list[ParsedPokemon] = field(default_factory=list)
    hand: list[Any] = field(default_factory=list)
    hand_count: int = 0
    deck_count: int = 0
    prize_remaining: int = 0
    discard: list[Any] = field(default_factory=list)
    bench_max: int = 0


@dataclass
class ParsedSelect:
    type: int | None
    context: int | None
    min_count: int
    max_count: int
    options: list[Any]
    deck: list[Any] | None = None
    context_card: Any | None = None
    effect: Any | None = None


@dataclass
class ParsedState:
    current_player: int
    turn: int
    turn_action_count: int
    result: int
    supporter_played: bool
    energy_attached: bool
    players: list[ParsedPlayer]
    select: ParsedSelect | None
    logs: list[Any]

    @property
    def me(self) -> ParsedPlayer:
        return self.players[self.current_player]

    @property
    def opp(self) -> ParsedPlayer:
        return self.players[1 - self.current_player]


class GameLedger:
    """Public-information ledger used by V0 rules and V1 belief sampling."""

    def __init__(self) -> None:
        self.seen_logs = 0
        self.public_counts: dict[int, int] = {}
        self.log_counts_by_type: dict[int, int] = {}

    def update(self, parsed: ParsedState) -> None:
        self.seen_logs += len(parsed.logs)
        for log in parsed.logs:
            log_type = enum_value(safe_get(log, "type"))
            if log_type is not None:
                self.log_counts_by_type[log_type] = self.log_counts_by_type.get(log_type, 0) + 1
        for player in parsed.players:
            for card in player.hand + player.discard:
                card_id = safe_get(card, "id")
                if card_id is not None:
                    self.public_counts[card_id] = self.public_counts.get(card_id, 0) + 1
            for pokemon in ([player.active] if player.active else []) + player.bench:
                self.public_counts[pokemon.id] = self.public_counts.get(pokemon.id, 0) + 1


def _parse_pokemon(raw: Any, area: int, index: int, player_index: int) -> ParsedPokemon | None:
    if raw is None:
        return None
    energies = safe_get(raw, "energies", []) or []
    energy_cards = safe_get(raw, "energyCards", []) or []
    tools = safe_get(raw, "tools", []) or []
    return ParsedPokemon(
        id=int(safe_get(raw, "id", 0) or 0),
        serial=int(safe_get(raw, "serial", 0) or 0),
        hp=int(safe_get(raw, "hp", 0) or 0),
        max_hp=int(safe_get(raw, "maxHp", safe_get(raw, "max_hp", 0)) or 0),
        area=area,
        index=index,
        player_index=player_index,
        energy_count=max(len(energies), len(energy_cards)),
        tool_count=len(tools),
        appear_this_turn=bool(safe_get(raw, "appearThisTurn", False)),
    )


def _parse_player(raw: Any, index: int) -> ParsedPlayer:
    active_raw = safe_get(raw, "active", []) or []
    active = _parse_pokemon(active_raw[0], enum_value(AreaType.ACTIVE), 0, index) if active_raw else None
    bench = [
        p
        for i, item in enumerate(safe_get(raw, "bench", []) or [])
        if (p := _parse_pokemon(item, enum_value(AreaType.BENCH), i, index)) is not None
    ]
    return ParsedPlayer(
        index=index,
        active=active,
        bench=bench,
        hand=list(safe_get(raw, "hand", []) or []),
        hand_count=int(safe_get(raw, "handCount", 0) or 0),
        deck_count=int(safe_get(raw, "deckCount", 0) or 0),
        prize_remaining=len(safe_get(raw, "prize", []) or []),
        discard=list(safe_get(raw, "discard", []) or []),
        bench_max=int(safe_get(raw, "benchMax", 0) or 0),
    )


def parse_observation(obs_dict: dict | Observation) -> ParsedState:
    obs: Observation = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
    current = obs.current
    if current is None:
        raise ValueError("deck selection observation has no current state")
    select = None
    if obs.select is not None:
        select = ParsedSelect(
            type=enum_value(obs.select.type),
            context=enum_value(obs.select.context),
            min_count=int(obs.select.minCount),
            max_count=int(obs.select.maxCount),
            options=list(obs.select.option or []),
            deck=obs.select.deck,
            context_card=obs.select.contextCard,
            effect=obs.select.effect,
        )
    players = [_parse_player(player, i) for i, player in enumerate(current.players)]
    return ParsedState(
        current_player=int(current.yourIndex),
        turn=int(current.turn),
        turn_action_count=int(current.turnActionCount),
        result=int(current.result),
        supporter_played=bool(current.supporterPlayed),
        energy_attached=bool(current.energyAttached),
        players=players,
        select=select,
        logs=list(obs.logs or []),
    )


def card_id_from_area(parsed: ParsedState, area: int | None, index: int | None, player_index: int | None) -> int | None:
    if area is None or index is None:
        return None
    player = parsed.players[player_index if player_index is not None else parsed.current_player]
    if area == enum_value(AreaType.HAND):
        if 0 <= index < len(player.hand):
            return safe_get(player.hand[index], "id")
    if area == enum_value(AreaType.DECK) and parsed.select and parsed.select.deck is not None:
        if 0 <= index < len(parsed.select.deck):
            return safe_get(parsed.select.deck[index], "id")
    if area == enum_value(AreaType.DISCARD):
        if 0 <= index < len(player.discard):
            return safe_get(player.discard[index], "id")
    if area == enum_value(AreaType.ACTIVE):
        return player.active.id if player.active else None
    if area == enum_value(AreaType.BENCH):
        if 0 <= index < len(player.bench):
            return player.bench[index].id
    if area == enum_value(AreaType.LOOKING) and parsed.select and parsed.select.deck is not None:
        # Some effects expose looked-at cards through select.deck while options use LOOKING.
        if 0 <= index < len(parsed.select.deck):
            return safe_get(parsed.select.deck[index], "id")
    return None


def target_pokemon(parsed: ParsedState, area: int | None, index: int | None, player_index: int | None) -> ParsedPokemon | None:
    if area is None or index is None:
        return None
    player = parsed.players[player_index if player_index is not None else parsed.current_player]
    if area == enum_value(AreaType.ACTIVE):
        return player.active
    if area == enum_value(AreaType.BENCH) and 0 <= index < len(player.bench):
        return player.bench[index]
    return None
