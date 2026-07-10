"""Heuristic leaf evaluation for root search."""

from __future__ import annotations

from cg.api import Observation

from .deck_profile_abomasnow import BASIC_WATER_ENERGY, KYOGRE, MEGA_ABOMASNOW_EX, SNOVER
from .parser import ParsedPlayer, ParsedState, parse_observation, safe_get


def _board_value(player: ParsedPlayer) -> float:
    value = 0.0
    for pokemon in ([player.active] if player.active else []) + player.bench:
        if pokemon.id == MEGA_ABOMASNOW_EX:
            value += 650 + pokemon.energy_count * 90 + pokemon.hp * 0.7
        elif pokemon.id == SNOVER:
            value += 250 + pokemon.energy_count * 60 + pokemon.hp * 0.4
        elif pokemon.id == KYOGRE:
            value += 180 + pokemon.energy_count * 55 + pokemon.hp * 0.35
        else:
            value += pokemon.hp * 0.25 + pokemon.energy_count * 25
    value += player.hand_count * 18
    value += player.deck_count * 6
    value -= max(0, 5 - player.deck_count) * 140
    value += sum(12 for card in player.discard if safe_get(card, "id") == BASIC_WATER_ENERGY)
    return value


def evaluate(parsed: ParsedState, root_player: int) -> float:
    if parsed.result != -1:
        if parsed.result == root_player:
            return 100000.0
        if parsed.result == 2:
            return 0.0
        return -100000.0

    me = parsed.players[root_player]
    opp = parsed.players[1 - root_player]
    prize_score = (opp.prize_remaining - me.prize_remaining) * 900
    board_score = _board_value(me) - _board_value(opp)
    active_threat = 0.0
    if me.active and opp.active:
        if me.active.id == MEGA_ABOMASNOW_EX and me.active.energy_count >= 2 and opp.active.hp <= 220:
            active_threat += 450
        if opp.active.id == MEGA_ABOMASNOW_EX and opp.active.energy_count >= 2 and me.active.hp <= 220:
            active_threat -= 450
    return prize_score + board_score + active_threat


def evaluate_observation(observation: Observation, root_player: int) -> float:
    return evaluate(parse_observation(observation), root_player)

