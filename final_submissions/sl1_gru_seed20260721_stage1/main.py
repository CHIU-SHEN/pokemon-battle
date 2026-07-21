from __future__ import annotations
import os
import sys

for _path in (os.getcwd(), "/kaggle_simulations/agent"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action

LEDGER = GameLedger()
POLICY = None
LAST_ACTION_SOURCE = "init"


def read_deck_csv():
    for root in (os.getcwd(), "/kaggle_simulations/agent"):
        path = os.path.join(root, "deck.csv")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as stream:
                deck = [int(line.strip()) for line in stream if line.strip()]
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("deck.csv")


def get_policy():
    global POLICY
    if POLICY is None:
        from gru_model_runtime import GRUPolicy
        POLICY = GRUPolicy(read_deck_csv(), window_length=16)
    return POLICY


def action_source():
    return LAST_ACTION_SOURCE


def agent(obs_dict):
    global LEDGER, LAST_ACTION_SOURCE
    if obs_dict is None or obs_dict.get("select") is None:
        if POLICY is not None:
            POLICY.reset()
        LEDGER = GameLedger()
        LAST_ACTION_SOURCE = "deck"
        return read_deck_csv()
    try:
        obs: Observation = to_observation_class(obs_dict)
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        try:
            action = get_policy().choose(parsed)
            if is_legal_action(obs.select, action):
                LAST_ACTION_SOURCE = "gru"
                return action
        except Exception:
            pass
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            LAST_ACTION_SOURCE = "rules_fallback"
            return action
        LAST_ACTION_SOURCE = "safe_fallback"
        return safe_action(obs.select, parsed, prefer_empty=False)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            LAST_ACTION_SOURCE = "exception_fallback"
            return safe_action(obs.select, prefer_empty=False) if obs.select is not None else read_deck_csv()
        except Exception:
            LAST_ACTION_SOURCE = "fatal_fallback"
            return []
