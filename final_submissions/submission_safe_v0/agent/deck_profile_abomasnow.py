"""Deck-specific constants and tunable weights for the V0 Abomasnow agent."""

from __future__ import annotations


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

