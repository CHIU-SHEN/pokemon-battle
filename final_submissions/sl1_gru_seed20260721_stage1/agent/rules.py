"""V0 card-specialized rule policy for the Mega Abomasnow deck."""

from __future__ import annotations

from typing import Any

from cg.api import OptionType, SelectContext

from .deck_profile_abomasnow import (
    ATTACK_DAMAGE,
    ATTACK_ENERGY_COST,
    BASIC_WATER_ENERGY,
    CARD_BASE_VALUE,
    CYRANO,
    DRAW_SUPPORTERS,
    ENERGY,
    HIGH_DAMAGE_ATTACKS,
    KYOGRE,
    LILLIES_DETERMINATION,
    MAIN_ATTACKERS,
    MAXIMUM_BELT,
    MEGA_ABOMASNOW_EX,
    MEGA_SIGNAL,
    POKEMON,
    SEARCH_CARDS,
    SETUP_BASICS,
    SNOVER,
    SUPPORTERS,
    TOOLS,
    WAITRESS,
    WEIGHTS,
)
from .fallback import is_legal_action, safe_action
from .parser import ParsedPokemon, ParsedState, card_id_from_area, enum_value, safe_get, target_pokemon


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
