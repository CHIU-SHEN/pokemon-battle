from __future__ import annotations

import json
from pathlib import Path
import tarfile


def test_builds_distinct_v3_s16_authority_and_kaggle_candidates(tmp_path: Path) -> None:
    from scripts.build_v3_s16_candidates import build_all
    from scripts.verify_v3_s16_candidate import verify

    root = Path(__file__).resolve().parents[1]
    result = build_all(tmp_path, code_root=root, frozen_root=root)

    authority = result["authority_archive"]
    kaggle = result["kaggle_archive"]
    assert authority.name == "pokemon-tcg-v3-s16-authority.tar.gz"
    assert kaggle.name == "pokemon-tcg-v3-s16-kaggle-60ms.tar.gz"
    assert authority != kaggle

    authority_root = tmp_path / "authority"
    kaggle_root = tmp_path / "kaggle"
    with tarfile.open(authority, "r:gz") as bundle:
        bundle.extractall(authority_root)
    with tarfile.open(kaggle, "r:gz") as bundle:
        bundle.extractall(kaggle_root)

    authority_package = authority_root / "pokemon-tcg-v3-s16-authority"
    authority_report = verify(authority_package, expected_variant="authority")
    kaggle_report = verify(kaggle_root, expected_variant="kaggle-60ms")
    assert authority_report["source_epoch"] == 44
    assert kaggle_report["source_epoch"] == 44

    authority_config = json.loads(
        (authority_package / "runtime_config.json").read_text(encoding="utf-8")
    )
    kaggle_config = json.loads(
        (kaggle_root / "runtime_config.json").read_text(encoding="utf-8")
    )
    assert authority_config == {
        "simulations": 16,
        "particles": 3,
        "max_depth": 10,
        "time_budget_seconds": 0.25,
        "game_budget_seconds": 120.0,
    }
    assert kaggle_config == {
        "simulations": 16,
        "particles": 3,
        "max_depth": 10,
        "time_budget_seconds": 0.06,
        "game_budget_seconds": 5.0,
    }
    assert (kaggle_root / "main.py").is_file()
    assert (kaggle_root / "deck.csv").is_file()
    assert (kaggle_root / "cg/api.py").is_file()
    assert not list(kaggle_root.rglob(".DS_Store"))
    manifest = json.loads((kaggle_root / "CANDIDATE_MANIFEST.json").read_text())
    assert manifest["formal_submission_replacement_authorized"] is False
    assert manifest["kaggle_upload_ready"] is False


def test_builds_flat_v3_s128_teacher_submission_candidate(tmp_path: Path) -> None:
    """Catch omission or mislabeling of the deployable 128-search teacher package."""
    from scripts.build_v3_s16_candidates import build_all
    from scripts.verify_v3_s16_candidate import verify

    root = Path(__file__).resolve().parents[1]
    result = build_all(tmp_path, code_root=root, frozen_root=root)

    archive = result["teacher_s128_archive"]
    assert archive.name == "pokemon-tcg-v3-teacher-s128.tar.gz"
    extracted = tmp_path / "teacher-s128"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted)

    report = verify(extracted, expected_variant="teacher-s128")
    assert report["source_epoch"] == 44
    assert json.loads((extracted / "runtime_config.json").read_text(encoding="utf-8")) == {
        "simulations": 128,
        "particles": 3,
        "max_depth": 10,
        "time_budget_seconds": 2.0,
        "game_budget_seconds": 120.0,
    }
    manifest = json.loads((extracted / "CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "teacher-s128"
    assert manifest["kaggle_upload_ready"] is False
