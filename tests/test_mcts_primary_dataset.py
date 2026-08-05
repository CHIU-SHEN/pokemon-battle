from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_game(root: Path, index: int, split: str) -> None:
    path = root / f"shards/w{index}/games/game_{index:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    game_id = f"run:primary:{index:06d}:side{index % 2}"
    path.write_text(json.dumps({
        "schema_version": "top2_mcts_game_v1", "game_id": game_id,
        "iteration_id": "run", "branch": "primary", "deck_id": "deck",
        "exceptions": [], "illegal_actions": [0, 0], "action_sources": {"mcts": 1},
        "samples": [{"game_id": game_id, "branch": "primary", "deck_id": "deck", "split": split,
                     "simulations": 2, "actions": [[0]], "visit_counts": [2], "policy_target": [1.0]}],
    }), encoding="utf-8")


def test_build_and_verify_frozen_dataset(tmp_path: Path) -> None:
    from scripts.build_mcts_primary_dataset import build
    from scripts.verify_mcts_primary_dataset import verify

    source = tmp_path / "source"
    for index, split in enumerate(("train", "valid", "test")):
        _write_game(source, index, split)
    archive, checksum, manifest = build(source, tmp_path / "out", identity={"branch": "primary", "deck_id": "deck"})
    assert manifest["schema_version"] == "mcts_primary_dataset_v2"
    assert manifest["totals"]["games"] == 3
    assert checksum.is_file()
    assert verify(archive)["totals"]["games"] == 3


def test_verifier_rejects_tampered_member_hash(tmp_path: Path) -> None:
    from scripts.build_mcts_primary_dataset import build
    from scripts.verify_mcts_primary_dataset import verify

    source = tmp_path / "source"
    for index, split in enumerate(("train", "valid", "test")):
        _write_game(source, index, split)
    archive, _, _ = build(source, tmp_path / "out", identity={"branch": "primary", "deck_id": "deck"})
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        verify(archive)
