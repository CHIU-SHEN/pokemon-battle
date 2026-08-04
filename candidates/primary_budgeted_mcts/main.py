"""Isolated primary Crustle candidate with budgeted belief-PUCT search."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


PRIMARY_CANDIDATE_ID = "crustle_kangaskhan_cage"
PRIMARY_DECK_ID = "top2-primary-crustle-kangaskhan-cage-v1"
PRIMARY_DECK_RELATIVE = Path("data/high_score_decks/crustle_kangaskhan_cage/deck.csv")
PRIMARY_DECK_CANDIDATES = (Path("deck.csv"), PRIMARY_DECK_RELATIVE)
_RUNTIME: Any | None = None


def find_project_root() -> Path:
    module_file = globals().get("__file__")
    starts = [Path.cwd(), Path("/kaggle_simulations/agent")]
    if module_file:
        starts.append(Path(module_file).resolve().parent)
    checked: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)
            if any((candidate / relative).is_file() for relative in PRIMARY_DECK_CANDIDATES):
                return candidate
    raise FileNotFoundError("primary deck assets not found")


def runtime_config_from_env() -> dict[str, int | float]:
    return {
        "simulations": int(os.environ.get("PTCG_MCTS_SIMULATIONS", "8")),
        "particles": int(os.environ.get("PTCG_MCTS_PARTICLES", "1")),
        "max_depth": int(os.environ.get("PTCG_MCTS_MAX_DEPTH", "4")),
        "time_budget_seconds": float(os.environ.get("PTCG_MCTS_TIME_BUDGET", "0.030")),
        "game_budget_seconds": float(os.environ.get("PTCG_MCTS_GAME_BUDGET", "2.0")),
    }


def read_primary_deck() -> list[int]:
    root = find_project_root()
    for relative in PRIMARY_DECK_CANDIDATES:
        path = root / relative
        if path.is_file():
            return [
                int(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    raise FileNotFoundError("primary deck assets not found")


def _get_runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    root = find_project_root()
    for import_root in (root, root / "submission"):
        value = str(import_root)
        if value not in sys.path:
            sys.path.insert(0, value)
    from src.rl.belief_puct_agent import SearchConfig, Top2BeliefPUCTAgent
    from src.rl.top2_rollout import Top2RolloutAgent

    settings = runtime_config_from_env()
    policy = Top2RolloutAgent(
        PRIMARY_CANDIDATE_ID,
        PRIMARY_DECK_ID,
        project_root=root,
        device=os.environ.get("PTCG_MCTS_DEVICE", "cpu"),
        deterministic=True,
        record_decisions=False,
    )
    _RUNTIME = Top2BeliefPUCTAgent(
        policy,
        config=SearchConfig(root_noise=False, **settings),
        selfplay=False,
    )
    return _RUNTIME


def agent(obs_dict: dict | None) -> list[int]:
    if obs_dict is None or obs_dict.get("select") is None:
        if _RUNTIME is not None:
            _RUNTIME.reset_trajectory()
        return read_primary_deck()
    runtime = _get_runtime()
    try:
        action = runtime(obs_dict)
        from agent.fallback import is_legal_action
        from cg.api import to_observation_class

        obs = to_observation_class(obs_dict)
        if obs.select is not None and is_legal_action(obs.select, action):
            return action
    except Exception:
        pass
    return runtime.policy(obs_dict)

