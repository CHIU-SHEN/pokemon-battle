"""Frozen V0 baseline: rules + fallback, no Search API."""

from __future__ import annotations

import os
import sys


BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASELINE_DIR))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
if SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, SUBMISSION_DIR)

from cg.api import Observation, to_observation_class  # noqa: E402
from agent.fallback import is_legal_action, safe_action  # noqa: E402
from agent.parser import GameLedger, parse_observation  # noqa: E402
from agent.rules import choose_action  # noqa: E402


LEDGER = GameLedger()


def read_deck_csv() -> list[int]:
    file_path = os.path.join(SUBMISSION_DIR, "deck.csv")
    with open(file_path, "r", encoding="utf-8") as file:
        return [int(line.strip()) for line in file if line.strip()]


def agent(obs_dict: dict | None) -> list[int]:
    if obs_dict is None:
        return read_deck_csv()
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
    except Exception:
        pass
    return safe_action(obs.select, prefer_empty=False)

