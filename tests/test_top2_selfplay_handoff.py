from __future__ import annotations

import json
from pathlib import Path
import tarfile


def test_handoff_contains_parallel_branch_jobs_and_no_training_data(tmp_path: Path) -> None:
    from scripts.build_top2_selfplay_handoff import build

    root = Path(__file__).resolve().parents[1]
    archive, checksum, manifest = build(tmp_path, root, root)

    assert archive.is_file()
    assert checksum.is_file()
    assert manifest["branches"] == ["primary", "reserve"]
    assert manifest["default_iterations"] == 1
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
    assert any(name.endswith("jobs/top2_selfplay_rollout.slurm") for name in names)
    assert any(name.endswith("jobs/top2_selfplay_train.slurm") for name in names)
    assert not any(name.endswith(".jsonl") for name in names)
    assert not any("/experiments/" in name for name in names)
    assert not any(name.endswith(".DS_Store") for name in names)


def test_manifest_declares_manual_review_before_four_more_rounds(tmp_path: Path) -> None:
    from scripts.build_top2_selfplay_handoff import build

    root = Path(__file__).resolve().parents[1]
    _, _, manifest = build(tmp_path, root, root)

    assert manifest["first_batch_iterations"] == 1
    assert manifest["continuation_iterations"] == 4
    assert manifest["submission_replacement_authorized"] is False
    assert json.dumps(manifest)
