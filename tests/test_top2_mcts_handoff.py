from __future__ import annotations

from pathlib import Path
import tempfile


def test_mcts_handoff_is_non_promoting_and_contains_single_node_job() -> None:
    from scripts.build_top2_mcts_handoff import build

    root = Path(__file__).resolve().parents[1]
    frozen = Path("E:/学校文件/kaggle/pokemon-battle")
    with tempfile.TemporaryDirectory(prefix="mcts_handoff_") as tmp:
        archive, checksum, manifest = build(Path(tmp), root, frozen)

        assert archive.is_file()
        assert checksum.is_file()
        assert manifest["submission_replacement_authorized"] is False
        assert manifest["pilot_games_per_branch"] == 200
        assert "jobs/top2_mcts_pilot_single_node.sh" in manifest["files"]
        assert not manifest["missing_frozen_files"]
