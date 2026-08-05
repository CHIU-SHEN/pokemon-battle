from __future__ import annotations

from pathlib import Path
import tarfile


def test_all_in_one_handoff_builds_verifies_and_has_lf_jobs(tmp_path: Path) -> None:
    from scripts.build_mcts_all_in_one_handoff import build
    from scripts.verify_mcts_all_in_one_handoff import verify

    root = Path(__file__).resolve().parents[1]
    frozen = Path("E:/学校文件/kaggle/pokemon-battle")
    archive, checksum, manifest = build(tmp_path, code_root=root, frozen_root=frozen)
    assert archive.name == "mcts-teacher-v2-all-in-one.tar.gz"
    assert checksum.is_file()
    assert not manifest["missing_frozen_files"]
    assert verify(archive)["schema_version"] == "mcts_teacher_v2_all_in_one_v1"
    with tarfile.open(archive, "r:gz") as bundle:
        job = bundle.extractfile("mcts-teacher-v2-all-in-one/jobs/mcts_teacher_v2_all_in_one.sh").read()
    assert b"\r" not in job
    assert job.index(b"run_worker_stage") < job.index(b"train_top2_mcts.py")
