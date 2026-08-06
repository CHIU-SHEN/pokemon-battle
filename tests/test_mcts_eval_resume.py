from __future__ import annotations

import json
from pathlib import Path

import pytest


def _identity(*, games: int = 4) -> dict:
    from scripts.evaluate_top2_mcts import evaluation_identity

    return evaluation_identity(
        branch="primary",
        kind="search",
        games=games,
        simulations=32,
        particles=3,
        max_depth=8,
        checkpoint=None,
    )


def test_progress_survives_interruption_and_preserves_side_parity(tmp_path: Path) -> None:
    from scripts.evaluate_top2_mcts import (
        load_progress,
        new_progress,
        record_completed_game,
    )

    progress_path = tmp_path / "search.progress.json"
    identity = _identity()
    progress = new_progress(identity)
    for game_index in range(2):
        record_completed_game(
            progress_path,
            progress,
            game_index=game_index,
            result=game_index % 2,
            tested_side=game_index % 2,
            exceptions=0,
            illegal_actions=0,
            fallbacks=0,
            decisions=1,
            latencies=[0.1],
        )

    assert json.loads(progress_path.read_text(encoding="utf-8"))["completed_games"] == 2

    resumed = load_progress(progress_path, identity, resume=True)
    resumed_sides = list(resumed["tested_sides"])
    for game_index in range(resumed["completed_games"], identity["games"]):
        resumed_sides.append(game_index % 2)
    assert resumed_sides == [0, 1, 0, 1]


def test_resume_rejects_incompatible_identity(tmp_path: Path) -> None:
    from scripts.evaluate_top2_mcts import load_progress, new_progress, atomic_write_json

    progress_path = tmp_path / "search.progress.json"
    atomic_write_json(progress_path, new_progress(_identity(games=4)))

    with pytest.raises(ValueError, match="does not match"):
        load_progress(progress_path, _identity(games=5), resume=True)


def test_non_resume_starts_fresh_even_when_progress_exists(tmp_path: Path) -> None:
    from scripts.evaluate_top2_mcts import load_progress, new_progress, atomic_write_json

    progress_path = tmp_path / "search.progress.json"
    progress = new_progress(_identity())
    progress["completed_games"] = 2
    atomic_write_json(progress_path, progress)

    fresh = load_progress(progress_path, _identity(), resume=False)
    assert fresh["completed_games"] == 0


def test_all_mcts_fallback_sources_are_counted() -> None:
    from scripts.evaluate_top2_mcts import count_mcts_fallbacks

    sources = {
        "mcts": 20,
        "policy_fallback": 3,
        "mcts_fallback": 1,
        "mcts_deadline_fallback": 2,
        "mcts_game_budget_fallback": 4,
        "mcts_exception_fallback": 8,
        "mcts_illegal_fallback": 16,
    }
    assert count_mcts_fallbacks(sources) == 31
