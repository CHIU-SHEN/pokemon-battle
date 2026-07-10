#!/usr/bin/env python3
"""M4 tactical regression tests built from synthetic observations."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))


def load_submission_agent():
    os.environ["PTCG_ENABLE_SEARCH"] = "0"
    path = SUBMISSION_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("submission_main_tactics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    os.chdir(SUBMISSION_DIR)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module.agent


def card(card_id: int, serial: int = 1, player: int = 0) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def pokemon(card_id: int, hp: int, max_hp: int, energies: int = 0, serial: int = 1, player: int = 0) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": player,
        "hp": hp,
        "maxHp": max_hp,
        "appearThisTurn": False,
        "energies": [3] * energies,
        "energyCards": [card(3, 100 + i, player) for i in range(energies)],
        "tools": [],
        "preEvolution": [],
    }


def player_state(
    idx: int,
    active: dict | None,
    bench: list[dict] | None = None,
    hand: list[dict] | None = None,
    deck_count: int = 30,
    prize_count: int = 6,
) -> dict:
    return {
        "active": [active] if active else [],
        "bench": bench or [],
        "benchMax": 5,
        "deckCount": deck_count,
        "discard": [],
        "prize": [None] * prize_count,
        "handCount": len(hand or []),
        "hand": hand or [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def observation(select: dict, p0: dict, p1: dict | None = None) -> dict:
    return {
        "select": select,
        "logs": [],
        "current": {
            "turn": 1,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                p0,
                p1 or player_state(1, pokemon(722, 90, 90, player=1), hand=None),
            ],
        },
        "search_begin_input": None,
    }


def select_data(context: int, options: list[dict], min_count: int = 1, max_count: int = 1, deck=None) -> dict:
    return {
        "type": 1 if context != 0 else 0,
        "context": context,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": options,
        "deck": deck,
        "contextCard": None,
        "effect": None,
    }


def require_action(name: str, got: list[int], expected: list[int]) -> None:
    if got != expected:
        raise AssertionError(f"{name}: expected {expected}, got {got}")


def main() -> int:
    agent = load_submission_agent()

    obs = observation(
        select_data(
            1,
            [
                {"type": 3, "area": 2, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 2, "index": 1, "playerIndex": 0},
            ],
        ),
        player_state(0, None, hand=[card(721, 1), card(722, 2)]),
    )
    require_action("setup active prefers Snover", agent(obs), [1])

    obs = observation(
        select_data(
            0,
            [
                {"type": 9, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": 14},
            ],
        ),
        player_state(0, pokemon(722, 90, 90, energies=2), hand=[card(723, 3)]),
    )
    require_action("evolve main attacker", agent(obs), [0])

    obs = observation(
        select_data(
            0,
            [
                {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
                {"type": 14},
            ],
        ),
        player_state(0, pokemon(723, 300, 350, energies=1), bench=[pokemon(721, 150, 150)], hand=[card(3, 4)]),
    )
    require_action("attach energy to ready attacker", agent(obs), [0])

    obs = observation(
        select_data(
            0,
            [
                {"type": 13, "attackId": 1046},
                {"type": 13, "attackId": 1047},
                {"type": 14},
            ],
        ),
        player_state(0, pokemon(723, 300, 350, energies=3), deck_count=5),
        player_state(1, pokemon(722, 190, 190, player=1), hand=None),
    )
    require_action("avoid deck-out Hammer-lanche", agent(obs), [1])

    deck_cards = [card(3, 10), card(723, 11), card(722, 12)]
    obs = observation(
        select_data(
            7,
            [
                {"type": 3, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 1, "playerIndex": 0},
                {"type": 3, "area": 1, "index": 2, "playerIndex": 0},
            ],
            deck=deck_cards,
        ),
        player_state(0, pokemon(722, 90, 90), hand=[]),
    )
    require_action("search finds Mega Abomasnow", agent(obs), [1])

    print("OK: tactical tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

