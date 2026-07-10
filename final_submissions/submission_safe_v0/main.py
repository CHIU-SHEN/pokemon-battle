import os

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action


LEDGER = GameLedger()


def read_deck_csv():
    for file_path in ("deck.csv", os.path.join(os.getcwd(), "deck.csv"), "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return [int(line.strip()) for line in file if line.strip()]
    raise FileNotFoundError("deck.csv not found")


def agent(obs_dict):
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
