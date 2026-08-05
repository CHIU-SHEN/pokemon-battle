from __future__ import annotations

import json
from pathlib import Path
import tarfile


def test_v3_handoff_builds_and_verifies_complete_quality_gated_archive(
    tmp_path: Path,
) -> None:
    from scripts.build_mcts_teacher_v3_handoff import build
    from scripts.verify_mcts_teacher_v3_handoff import verify

    root = Path(__file__).resolve().parents[1]
    archive, checksum, manifest = build(tmp_path, code_root=root, frozen_root=root)

    assert archive.name == "mcts-teacher-v3-quality-gated.tar.gz"
    assert checksum.name == "mcts-teacher-v3-quality-gated.tar.gz.sha256"
    assert manifest["schema_version"] == "mcts_teacher_v3_quality_gated_v1"
    assert manifest["missing_frozen_files"] == []
    assert manifest["defaults"] == {
        "teacher_gate_games": 400,
        "teacher_min_win_rate": 0.58,
        "target_games": 5000,
        "simulations": 128,
        "particles": 3,
        "max_depth": 10,
        "workers": 16,
    }
    assert verify(archive)["verified"] is True

    package = "mcts-teacher-v3-quality-gated"
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        job = bundle.extractfile(
            f"{package}/jobs/mcts_teacher_v3_quality_gated.sh"
        ).read()
        archived_manifest = json.load(bundle.extractfile(f"{package}/HANDOFF_MANIFEST.json"))
    assert b"\r" not in job
    assert f"{package}/docs/MCTS_TEACHER_V3_SERVER_RUNBOOK.md" in names
    assert f"{package}/docs/MCTS_HYBRID_POLICY_DECISION.md" in names
    assert f"{package}/scripts/gate_mcts_teacher.py" in names
    assert f"{package}/scripts/verify_mcts_teacher_v3_handoff.py" in names
    assert not any(name.endswith(".DS_Store") for name in names)
    assert archived_manifest["files"]
