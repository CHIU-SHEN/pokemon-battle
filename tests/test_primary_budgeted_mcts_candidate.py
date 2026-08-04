from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MAIN = ROOT / "candidates/primary_budgeted_mcts/main.py"


def _module():
    spec = importlib.util.spec_from_file_location("primary_budgeted_mcts_main", CANDIDATE_MAIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_uses_safe_budget_defaults(monkeypatch) -> None:
    module = _module()
    for name in (
        "PTCG_MCTS_SIMULATIONS",
        "PTCG_MCTS_PARTICLES",
        "PTCG_MCTS_MAX_DEPTH",
        "PTCG_MCTS_TIME_BUDGET",
        "PTCG_MCTS_GAME_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)

    config = module.runtime_config_from_env()
    assert config == {
        "simulations": 8,
        "particles": 1,
        "max_depth": 4,
        "time_budget_seconds": 0.030,
        "game_budget_seconds": 2.0,
    }


def test_candidate_returns_exact_primary_deck_without_loading_model() -> None:
    module = _module()
    expected = [
        int(line)
        for line in (ROOT / "data/high_score_decks/crustle_kangaskhan_cage/deck.csv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert module.agent(None) == expected
    assert module._RUNTIME is None


def test_candidate_is_isolated_from_formal_submission() -> None:
    module = _module()
    project_root = module.find_project_root()

    assert project_root == ROOT
    assert "submission" not in CANDIDATE_MAIN.relative_to(ROOT).parts
    assert module.PRIMARY_CANDIDATE_ID == "crustle_kangaskhan_cage"
