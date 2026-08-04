from __future__ import annotations

import json
import os
from pathlib import Path
import tarfile


def test_teacher_handoff_contains_frozen_inputs_and_resilient_workflow(tmp_path: Path) -> None:
    from scripts.build_mcts_teacher_v2_handoff import build

    root = Path(__file__).resolve().parents[1]
    frozen = Path(os.environ["PTCG_FROZEN_SOURCE_ROOT"])
    archive, checksum, manifest = build(tmp_path, code_root=root, frozen_root=frozen)

    assert archive.name == "mcts-distill-v2-teacher.tar.gz"
    assert checksum.is_file()
    assert manifest["schema_version"] == "mcts_teacher_v2_handoff_v1"
    assert manifest["branch"] == "primary"
    assert manifest["submission_replacement_authorized"] is False
    assert manifest["authoritative_archive_sha256"] == (
        "f926fbe822d18321d3e083bd30fd60a73da6f35517327f69d0e7bd44262cb531"
    )
    assert not manifest["missing_frozen_files"]
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        prefix = "mcts-distill-v2-teacher/"
        assert prefix + "src/rl/mcts_teacher.py" in names
        assert prefix + "scripts/train_top2_mcts.py" in names
        assert prefix + "scripts/run_mcts_teacher_smoke.py" in names
        assert prefix + "jobs/mcts_teacher_v2_resilient.sh" in names
        assert prefix + "docs/operations/MCTS_TEACHER_V2_SERVER_HANDOFF.md" in names
        assert prefix + "artifacts/top2-mcts-complete-results-20260804.tar.gz" in names
        manifest_doc = json.load(bundle.extractfile(prefix + "HANDOFF_MANIFEST.json"))
    assert manifest_doc["files"] == manifest["files"]
