from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

for _path in (os.getcwd(), "/kaggle_simulations/agent"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import AreaType, Observation, OptionType, SelectContext, to_observation_class



# ---- inlined from agent/deck_profile_abomasnow.py ----

BASIC_WATER_ENERGY = 3
KYOGRE = 721
SNOVER = 722
MEGA_ABOMASNOW_EX = 723
MEGA_SIGNAL = 1145
MAXIMUM_BELT = 1158
CYRANO = 1205
LILLIES_DETERMINATION = 1227
WAITRESS = 1235

MAIN_ATTACKERS = {MEGA_ABOMASNOW_EX}
BASIC_POKEMON = {SNOVER, KYOGRE}
SETUP_BASICS = {SNOVER}
POKEMON = {KYOGRE, SNOVER, MEGA_ABOMASNOW_EX}
ENERGY = {BASIC_WATER_ENERGY}
SEARCH_CARDS = {MEGA_SIGNAL, CYRANO}
DRAW_SUPPORTERS = {LILLIES_DETERMINATION, WAITRESS}
SUPPORTERS = {CYRANO, LILLIES_DETERMINATION, WAITRESS}
TOOLS = {MAXIMUM_BELT}

CARD_NAMES = {
    BASIC_WATER_ENERGY: "Basic Water Energy",
    KYOGRE: "Kyogre",
    SNOVER: "Snover",
    MEGA_ABOMASNOW_EX: "Mega Abomasnow ex",
    MEGA_SIGNAL: "Mega Signal",
    MAXIMUM_BELT: "Maximum Belt",
    CYRANO: "Cyrano",
    LILLIES_DETERMINATION: "Lillie's Determination",
    WAITRESS: "Waitress",
}

CARD_BASE_VALUE = {
    MEGA_ABOMASNOW_EX: 950,
    SNOVER: 850,
    BASIC_WATER_ENERGY: 620,
    MEGA_SIGNAL: 560,
    CYRANO: 520,
    WAITRESS: 500,
    LILLIES_DETERMINATION: 470,
    MAXIMUM_BELT: 430,
    KYOGRE: 360,
}

ATTACK_DAMAGE = {
    1042: 0,    # Kyogre Riptide, dynamic
    1043: 130,  # Kyogre Swirling Waves
    1044: 10,   # Snover Beat
    1045: 30,   # Snover Icy Snow
    1046: 300,  # Mega Abomasnow Hammer-lanche rough expectation
    1047: 200,  # Mega Abomasnow Frost Barrier
}

ATTACK_ENERGY_COST = {
    1042: 1,
    1043: 3,
    1044: 1,
    1045: 2,
    1046: 2,
    1047: 3,
}

HIGH_DAMAGE_ATTACKS = {1046, 1047}

WEIGHTS = {
    "take_prize": 1000,
    "ko_ex_bonus": 500,
    "setup_main_attacker": 300,
    "attach_to_ready_attacker": 250,
    "avoid_deckout": 800,
    "waste_supporter_penalty": -200,
    "play_search": 380,
    "play_draw": 260,
    "play_tool": 180,
    "evolve_main_attacker": 720,
    "attack": 650,
    "end_turn": -150,
    "bench_snover": 420,
    "bench_kyogre": 80,
    "prefer_active_attacker": 180,
    "prefer_bench_attacker": 130,
    "low_value_discard_penalty": -120,
    "deckout_per_missing_card": -160,
}


# ---- inlined from agent/parser.py ----

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


# ---- inlined from agent/fallback.py ----

def _option_card_value(parsed: ParsedState | None, option: Any) -> int:
    direct = safe_get(option, "cardId")
    if direct is not None:
        return CARD_BASE_VALUE.get(int(direct), 0)
    if parsed is None:
        return 0
    card_id = card_id_from_area(
        parsed,
        enum_value(safe_get(option, "area")),
        safe_get(option, "index"),
        safe_get(option, "playerIndex"),
    )
    if card_id is None:
        card_id = card_id_from_area(
            parsed,
            enum_value(safe_get(option, "inPlayArea")),
            safe_get(option, "inPlayIndex"),
            parsed.current_player,
        )
    return CARD_BASE_VALUE.get(int(card_id), 0) if card_id is not None else 0


def _select_attr(select: Any, camel: str, snake: str, default: Any = None) -> Any:
    return safe_get(select, camel, safe_get(select, snake, default))


def safe_action(select: Any, parsed: ParsedState | None = None, prefer_empty: bool = True) -> list[int]:
    """Return a legal action for any official SelectData-like object.

    The fallback is deliberately conservative: optional unknown selections return
    an empty list, while mandatory selections choose the highest known card-value
    options and otherwise the first legal indices.
    """

    min_count = int(_select_attr(select, "minCount", "min_count", 0) or 0)
    max_count = int(_select_attr(select, "maxCount", "max_count", min_count) or min_count)
    options = list(_select_attr(select, "option", "options", []) or [])

    if min_count == 0 and prefer_empty:
        return []
    if max_count <= 0 or not options:
        return []

    ranked = sorted(
        range(len(options)),
        key=lambda i: (_option_card_value(parsed, options[i]), -i),
        reverse=True,
    )
    count = min(max(min_count, 0), max_count, len(options))
    return ranked[:count]


def is_legal_action(select: Any, action: list[int]) -> bool:
    min_count = int(_select_attr(select, "minCount", "min_count", 0) or 0)
    max_count = int(_select_attr(select, "maxCount", "max_count", min_count) or min_count)
    options = list(_select_attr(select, "option", "options", []) or [])
    return (
        isinstance(action, list)
        and all(isinstance(x, int) for x in action)
        and min_count <= len(action) <= max_count
        and len(set(action)) == len(action)
        and all(0 <= x < len(options) for x in action)
    )


# ---- inlined from agent/rules.py ----

def _oval(option: Any, name: str, default=None):
    return safe_get(option, name, default)


def _otype(option: Any) -> int | None:
    return enum_value(_oval(option, "type"))


def _has_card(cards: list[Any], card_id: int) -> bool:
    return any(safe_get(card, "id") == card_id for card in cards)


def _count_card(cards: list[Any], card_id: int) -> int:
    return sum(1 for card in cards if safe_get(card, "id") == card_id)


def _pokemon_readiness(pokemon: ParsedPokemon | None) -> int:
    if pokemon is None:
        return -1000
    if pokemon.id == MEGA_ABOMASNOW_EX:
        return 600 + pokemon.energy_count * 120 + pokemon.hp // 4
    if pokemon.id == SNOVER:
        return 360 + pokemon.energy_count * 80 + pokemon.hp // 6
    if pokemon.id == KYOGRE:
        return 260 + pokemon.energy_count * 90 + pokemon.hp // 8
    return pokemon.energy_count * 50 + pokemon.hp // 10


def _best_energy_target(parsed: ParsedState) -> ParsedPokemon | None:
    candidates = ([parsed.me.active] if parsed.me.active else []) + parsed.me.bench
    return max(candidates, key=_pokemon_readiness, default=None)


def _opponent_active_hp(parsed: ParsedState) -> int | None:
    return parsed.opp.active.hp if parsed.opp.active else None


def _is_ex(card_id: int | None) -> bool:
    return card_id == MEGA_ABOMASNOW_EX


def _damage_for_attack(parsed: ParsedState, attack_id: int | None) -> int:
    if attack_id is None:
        return 0
    damage = ATTACK_DAMAGE.get(int(attack_id), 0)
    if attack_id == 1042:
        # Kyogre Riptide: 20 per water energy in discard.
        damage = 20 * _count_card(parsed.me.discard, BASIC_WATER_ENERGY)
    if attack_id == 1046:
        # Hammer-lanche is risky when deck is thin; lower the expectation late.
        damage = min(600, max(0, parsed.me.deck_count - 2) * 42)
    return damage


def _score_attack(parsed: ParsedState, option: Any) -> int:
    attack_id = _oval(option, "attackId")
    damage = _damage_for_attack(parsed, attack_id)
    score = WEIGHTS["attack"] + damage
    opp_hp = _opponent_active_hp(parsed)
    if opp_hp is not None and damage >= opp_hp:
        score += WEIGHTS["take_prize"]
        if _is_ex(parsed.opp.active.id if parsed.opp.active else None):
            score += WEIGHTS["ko_ex_bonus"]
    if attack_id in HIGH_DAMAGE_ATTACKS:
        score += 120
    if attack_id == 1046 and parsed.me.deck_count <= 7:
        score -= WEIGHTS["avoid_deckout"]
        score += (8 - parsed.me.deck_count) * WEIGHTS["deckout_per_missing_card"]
    if attack_id == 1043:
        score -= 80  # discarding 2 energy slows the main plan.
    return score


def _score_play_card(parsed: ParsedState, card_id: int | None) -> int:
    if card_id is None:
        return 0
    me = parsed.me
    has_snover_in_play = any(p.id == SNOVER for p in ([me.active] if me.active else []) + me.bench)
    has_aboma_in_play = any(p.id == MEGA_ABOMASNOW_EX for p in ([me.active] if me.active else []) + me.bench)
    hand_has_aboma = _has_card(me.hand, MEGA_ABOMASNOW_EX)
    bench_open = me.bench_max == 0 or len(me.bench) < me.bench_max

    score = CARD_BASE_VALUE.get(card_id, 0)
    if card_id == SNOVER:
        score += WEIGHTS["setup_main_attacker"] + (WEIGHTS["bench_snover"] if bench_open else -300)
        if not has_snover_in_play:
            score += 450
    elif card_id == KYOGRE:
        score += WEIGHTS["bench_kyogre"] if bench_open else -260
    elif card_id in SEARCH_CARDS:
        score += WEIGHTS["play_search"]
        if not hand_has_aboma and has_snover_in_play and not has_aboma_in_play:
            score += 430
        if card_id == CYRANO and parsed.supporter_played:
            score += WEIGHTS["waste_supporter_penalty"]
    elif card_id in DRAW_SUPPORTERS:
        score += WEIGHTS["play_draw"]
        if parsed.supporter_played:
            score += WEIGHTS["waste_supporter_penalty"]
        if card_id == LILLIES_DETERMINATION and me.prize_remaining == 6:
            score += 220
        if card_id == WAITRESS and not parsed.energy_attached:
            score += 130
    elif card_id == MAXIMUM_BELT:
        score += WEIGHTS["play_tool"] if has_aboma_in_play or has_snover_in_play else -140
    elif card_id == BASIC_WATER_ENERGY:
        score -= 200
    return score


def _score_attach(parsed: ParsedState, option: Any) -> int:
    card_id = card_id_from_area(
        parsed,
        enum_value(_oval(option, "area")),
        _oval(option, "index"),
        parsed.current_player,
    )
    target = target_pokemon(
        parsed,
        enum_value(_oval(option, "inPlayArea")),
        _oval(option, "inPlayIndex"),
        parsed.current_player,
    )
    if card_id not in ENERGY and card_id not in TOOLS:
        return 0
    score = 0
    if card_id in ENERGY:
        score += WEIGHTS["attach_to_ready_attacker"]
        if target is not None:
            score += _pokemon_readiness(target)
            cost = 2 if target.id == MEGA_ABOMASNOW_EX else 1 if target.id == SNOVER else 3
            if target.energy_count < cost:
                score += 220
            if target == parsed.me.active:
                score += WEIGHTS["prefer_active_attacker"]
            else:
                score += WEIGHTS["prefer_bench_attacker"]
            if parsed.me.active and parsed.me.active.hp <= 80 and target != parsed.me.active:
                score += 160
    if card_id in TOOLS:
        if target and target.id in MAIN_ATTACKERS:
            score += 520
        elif target and target.id == SNOVER:
            score += 220
        else:
            score -= 140
    return score


def _score_evolve(parsed: ParsedState, option: Any) -> int:
    card_id = card_id_from_area(
        parsed,
        enum_value(_oval(option, "area")),
        _oval(option, "index"),
        parsed.current_player,
    )
    target = target_pokemon(
        parsed,
        enum_value(_oval(option, "inPlayArea")),
        _oval(option, "inPlayIndex"),
        parsed.current_player,
    )
    if card_id == MEGA_ABOMASNOW_EX and target and target.id == SNOVER:
        score = WEIGHTS["evolve_main_attacker"] + target.energy_count * 150 + target.hp
        if target == parsed.me.active:
            score += 120
        return score
    return CARD_BASE_VALUE.get(card_id, 0)


def _score_main_option(parsed: ParsedState, option: Any) -> int:
    option_type = _otype(option)
    if option_type == enum_value(OptionType.PLAY):
        card_id = card_id_from_area(parsed, enum_value(2), _oval(option, "index"), parsed.current_player)
        return _score_play_card(parsed, card_id)
    if option_type == enum_value(OptionType.ATTACH):
        return _score_attach(parsed, option)
    if option_type == enum_value(OptionType.EVOLVE):
        return _score_evolve(parsed, option)
    if option_type == enum_value(OptionType.ATTACK):
        return _score_attack(parsed, option)
    if option_type == enum_value(OptionType.RETREAT):
        return 60 if parsed.me.active and parsed.me.active.hp <= 60 and parsed.me.bench else -60
    if option_type == enum_value(OptionType.END):
        return WEIGHTS["end_turn"]
    if option_type == enum_value(OptionType.ABILITY):
        return 180
    if option_type == enum_value(OptionType.DISCARD):
        return WEIGHTS["low_value_discard_penalty"]
    return 0


def _score_card_choice(parsed: ParsedState, option: Any, context: int | None) -> int:
    area = enum_value(_oval(option, "area"))
    index = _oval(option, "index")
    player_index = _oval(option, "playerIndex", parsed.current_player)
    card_id = card_id_from_area(parsed, area, index, player_index)
    score = CARD_BASE_VALUE.get(card_id, 0) if card_id is not None else 0
    me = parsed.me

    if context == enum_value(SelectContext.SETUP_ACTIVE_POKEMON):
        if card_id == SNOVER:
            score += 900
        elif card_id == KYOGRE:
            score += 350
    elif context == enum_value(SelectContext.SETUP_BENCH_POKEMON):
        if card_id == SNOVER:
            score += 650
        elif card_id == KYOGRE:
            score += 100
    elif context in {enum_value(SelectContext.TO_BENCH), enum_value(SelectContext.TO_FIELD)}:
        if card_id == SNOVER:
            score += 650
        elif card_id == KYOGRE:
            score += 120
    elif context == enum_value(SelectContext.TO_ACTIVE):
        pokemon = target_pokemon(parsed, area, index, player_index)
        score += _pokemon_readiness(pokemon)
    elif context == enum_value(SelectContext.TO_HAND):
        has_snover = any(p.id == SNOVER for p in ([me.active] if me.active else []) + me.bench)
        if card_id == MEGA_ABOMASNOW_EX and has_snover:
            score += 900
        elif card_id == SNOVER and not has_snover:
            score += 760
        elif card_id == BASIC_WATER_ENERGY:
            score += 260
    elif context in {enum_value(SelectContext.ATTACH_FROM), enum_value(SelectContext.ATTACH_TO)}:
        if card_id == BASIC_WATER_ENERGY:
            score += 500
        pokemon = target_pokemon(parsed, area, index, player_index)
        if pokemon:
            score += _pokemon_readiness(pokemon)
    elif context in {
        enum_value(SelectContext.DISCARD),
        enum_value(SelectContext.TO_DECK),
        enum_value(SelectContext.TO_DECK_BOTTOM),
    }:
        score = -score
        if card_id == BASIC_WATER_ENERGY:
            score += 100  # Water in discard powers Kyogre and is less bad than losing attackers.
        if card_id in MAIN_ATTACKERS or card_id == SNOVER:
            score -= 700
    return score


def _ranked_action(parsed: ParsedState, scores: list[int], prefer_empty: bool = False) -> list[int]:
    select = parsed.select
    if select is None:
        return []
    if select.min_count == 0 and prefer_empty and max(scores, default=-10**9) <= 0:
        return []
    count = select.min_count
    if select.max_count > select.min_count:
        positive = sum(1 for score in scores if score > 0)
        count = min(select.max_count, max(select.min_count, positive))
    ranked = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)
    return ranked[:count]


def score_options(parsed: ParsedState) -> list[int]:
    select = parsed.select
    if select is None:
        return []
    if select.context == enum_value(SelectContext.MAIN):
        return [_score_main_option(parsed, option) for option in select.options]
    return [_score_card_choice(parsed, option, select.context) for option in select.options]


def choose_action(parsed: ParsedState) -> list[int]:
    select = parsed.select
    if select is None:
        return []

    try:
        context = select.context
        if context == enum_value(SelectContext.MAIN):
            scores = score_options(parsed)
            action = _ranked_action(parsed, scores)
        elif context == enum_value(SelectContext.IS_FIRST):
            # This deck wants the first manual attachment and evolution tempo.
            yes_indices = [i for i, option in enumerate(select.options) if _otype(option) == enum_value(OptionType.YES)]
            action = yes_indices[:1] if yes_indices else safe_action(select, parsed, prefer_empty=False)
        else:
            scores = score_options(parsed)
            action = _ranked_action(parsed, scores, prefer_empty=True)

        if is_legal_action(select, action):
            return action
    except Exception:
        pass

    return safe_action(select, parsed, prefer_empty=False)



LEDGER = GameLedger()


def read_deck_csv():
    for file_path in (
        "deck.csv",
        os.path.join(os.getcwd(), "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    return [
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


def agent(obs_dict):
    if obs_dict is None:
        return read_deck_csv()
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
        return safe_action(obs.select, parsed, prefer_empty=False)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return read_deck_csv()
            return safe_action(obs.select, prefer_empty=False)
        except Exception:
            return []
