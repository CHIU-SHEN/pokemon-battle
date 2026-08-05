from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rl.mcts_collection import audit_collection, choose_worker_candidates, plan_shards


def _game(path: Path, game_id: str, *, split: str = "train", **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "top2_mcts_game_v1",
        "game_id": game_id,
        "iteration_id": "run",
        "branch": "primary",
        "deck_id": "deck",
        "exceptions": [],
        "illegal_actions": [0, 0],
        "action_sources": {"mcts": 3},
        "samples": [{"game_id": game_id, "branch": "primary", "deck_id": "deck", "split": split, "simulations": 4, "actions": [[0]], "visit_counts": [4], "policy_target": [1.0]}],
        **extra,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_worker_candidates_are_conservative_and_override_is_exact() -> None:
    assert choose_worker_candidates(48) == (12, 16, 20)
    assert choose_worker_candidates(16) == (4,)
    assert choose_worker_candidates(48, override=7) == (7,)
    with pytest.raises(ValueError, match="positive"):
        choose_worker_candidates(48, override=0)


def test_shards_have_disjoint_identity_seed_path_and_even_games() -> None:
    plans = plan_shards(10, 3, seed=100, iteration_id="bench", completed_game_ids=set())
    assert [plan.games for plan in plans] == [4, 3, 3]
    assert len({plan.seed for plan in plans}) == 3
    assert len({plan.iteration_id for plan in plans}) == 3
    assert len({plan.relative_root for plan in plans}) == 3
    assert len({game_id for plan in plans for game_id in plan.game_ids}) == 10


def test_resume_excludes_completed_game_ids() -> None:
    completed = {"bench:primary:000000:side0", "bench:primary:000003:side1"}
    plans = plan_shards(6, 2, seed=10, iteration_id="bench", completed_game_ids=completed)
    scheduled = {game_id for plan in plans for game_id in plan.game_ids}
    assert scheduled.isdisjoint(completed)
    assert len(scheduled) == 4


def test_audit_rebuilds_cumulative_totals_and_action_sources(tmp_path: Path) -> None:
    _game(tmp_path / "a/games/game_0.json", "a", split="train")
    _game(tmp_path / "b/games/game_1.json", "b", split="valid")
    _game(tmp_path / "c/games/game_2.json", "c", split="test")
    report = audit_collection(tmp_path, {"branch": "primary", "deck_id": "deck"}, require_all_splits=True)
    assert report["games"] == 3
    assert report["samples"] == 3
    assert report["nodes"] == 12
    assert report["action_sources"] == {"mcts": 9}
    assert report["fallbacks"] == 0
    assert report["fallback_rate"] == 0.0


def test_audit_rejects_duplicate_identity_and_hidden_fields(tmp_path: Path) -> None:
    _game(tmp_path / "a/games/game.json", "duplicate")
    _game(tmp_path / "b/games/game.json", "duplicate")
    with pytest.raises(ValueError, match="duplicate game_id"):
        audit_collection(tmp_path, {"branch": "primary", "deck_id": "deck"})
    (tmp_path / "b/games/game.json").unlink()
    _game(tmp_path / "c/games/game.json", "hidden", opponent_hand=[1])
    with pytest.raises(ValueError, match="hidden belief"):
        audit_collection(tmp_path, {"branch": "primary", "deck_id": "deck"})


def test_audit_rejects_fallback_and_identity_mismatch(tmp_path: Path) -> None:
    _game(tmp_path / "a/games/game.json", "bad", action_sources={"mcts": 2, "mcts_exception_fallback": 1})
    with pytest.raises(ValueError, match="fallback"):
        audit_collection(tmp_path, {"branch": "primary", "deck_id": "deck"})
    _game(tmp_path / "a/games/game.json", "bad", branch="reserve")
    with pytest.raises(ValueError, match="branch"):
        audit_collection(tmp_path, {"branch": "primary", "deck_id": "deck"})
