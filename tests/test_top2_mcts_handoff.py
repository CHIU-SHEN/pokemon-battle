from __future__ import annotations

from pathlib import Path
import tempfile


def test_runner_resumes_both_arena_evaluations() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_top2_mcts_pilot.py").read_text(encoding="utf-8")

    assert runner.count('"--resume"') >= 3


def test_resilient_job_runs_branches_sequentially() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs/top2_mcts_pilot_resilient.sh").read_text(encoding="utf-8")

    assert 'for BRANCH in primary reserve' in job
    assert 'wait "$PRIMARY_PID"' not in job
    assert '--resume' in job


def test_mcts_handoff_is_non_promoting_and_contains_single_node_job() -> None:
    from scripts.build_top2_mcts_handoff import build

    root = Path(__file__).resolve().parents[1]
    frozen = root.parents[1]
    with tempfile.TemporaryDirectory(prefix="mcts_handoff_") as tmp:
        archive, checksum, manifest = build(Path(tmp), root, frozen)

        assert archive.is_file()
        assert archive.name == "pokemon-tcg-top2-mcts-pilot-v2.tar.gz"
        assert checksum.is_file()
        assert manifest["submission_replacement_authorized"] is False
        assert manifest["pilot_games_per_branch"] == 200
        assert "jobs/top2_mcts_pilot_single_node.sh" in manifest["files"]
        assert "jobs/top2_mcts_pilot_resilient.sh" in manifest["files"]
        assert not manifest["missing_frozen_files"]
