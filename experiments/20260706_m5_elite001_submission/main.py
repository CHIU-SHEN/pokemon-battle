import os

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action
from agent.search import SearchConfig, SearchManager


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER = GameLedger()
SEARCH_MANAGER = None


def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = os.path.join(AGENT_DIR, "deck.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def get_search_manager() -> SearchManager:
    global SEARCH_MANAGER
    if SEARCH_MANAGER is None:
        enabled = os.environ.get("PTCG_ENABLE_SEARCH", "0") not in {"0", "false", "False"}
        SEARCH_MANAGER = SearchManager(
            deck=read_deck_csv(),
            config=SearchConfig(
                enabled=enabled,
                max_candidates=int(os.environ.get("PTCG_SEARCH_CANDIDATES", "6")),
                particles=int(os.environ.get("PTCG_SEARCH_PARTICLES", "3")),
                node_budget=int(os.environ.get("PTCG_SEARCH_NODE_BUDGET", "64")),
                time_budget_sec=float(os.environ.get("PTCG_SEARCH_TIME_BUDGET", "0.035")),
                switch_margin=float(os.environ.get("PTCG_SEARCH_SWITCH_MARGIN", "175")),
            ),
        )
    return SEARCH_MANAGER

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    if obs_dict is None:
        return read_deck_csv()

    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()
    
    try:
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        action = get_search_manager().choose(obs_dict, parsed, LEDGER)
        if is_legal_action(obs.select, action):
            return action
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
    except Exception:
        pass

    return safe_action(obs.select, prefer_empty=False)
