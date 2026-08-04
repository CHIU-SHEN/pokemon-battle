from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile


def test_kaggle_archive_has_required_files_at_root(tmp_path: Path) -> None:
    from scripts.build_primary_budgeted_mcts_kaggle import PACKAGE_BASENAME, build

    code_root = Path(__file__).resolve().parents[1]
    frozen_root = Path(os.environ.get("PTCG_FROZEN_SOURCE_ROOT", code_root))

    archive, checksum, manifest = build(
        tmp_path, code_root=code_root, frozen_root=frozen_root
    )

    assert archive.name == f"{PACKAGE_BASENAME}.tar.gz"
    assert archive.is_file() and checksum.is_file()
    assert manifest["candidate_id"] == "crustle_kangaskhan_cage"
    assert manifest["kaggle_upload_ready"] is True
    assert not manifest["missing_frozen_files"]

    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    assert "main.py" in names
    assert "deck.csv" in names
    assert "cg/api.py" in names
    assert "KAGGLE_MANIFEST.json" in names
    assert "artifacts/sl0_shared_full/best.pt" in names
    assert "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt" in names
    assert not any(name.startswith(f"{PACKAGE_BASENAME}/") for name in names)
    assert not any("petrel" in name or "reserve" in name for name in names)
    assert not any("samples/games" in name or name.endswith(".jsonl") for name in names)


def test_verifier_checks_hashes_and_top_level_deck(tmp_path: Path) -> None:
    from scripts.build_primary_budgeted_mcts_kaggle import build
    from scripts.verify_primary_budgeted_mcts_kaggle import verify

    code_root = Path(__file__).resolve().parents[1]
    frozen_root = Path(os.environ.get("PTCG_FROZEN_SOURCE_ROOT", code_root))
    archive, _, _ = build(tmp_path, code_root=code_root, frozen_root=frozen_root)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted)

    result = verify(extracted)

    assert result == {
        "ok": True,
        "candidate_id": "crustle_kangaskhan_cage",
        "deck_cards": 60,
        "verified_files": 39,
    }

    (extracted / "deck.csv").write_text("1\n", encoding="utf-8")
    try:
        verify(extracted)
    except ValueError as exc:
        assert "hash failures" in str(exc)
    else:
        raise AssertionError("modified top-level deck must fail verification")


def test_builder_cli_runs_from_project_root() -> None:
    code_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/build_primary_budgeted_mcts_kaggle.py", "--help"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_packaged_agent_supports_kaggle_raw_exec_without_file(
    tmp_path: Path, monkeypatch,
) -> None:
    from scripts.build_primary_budgeted_mcts_kaggle import build

    code_root = Path(__file__).resolve().parents[1]
    frozen_root = Path(os.environ.get("PTCG_FROZEN_SOURCE_ROOT", code_root))
    archive, _, _ = build(tmp_path, code_root=code_root, frozen_root=frozen_root)
    extracted = tmp_path / "raw-exec"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted)

    monkeypatch.chdir(extracted)
    environment = {"__builtins__": __builtins__}
    exec((extracted / "main.py").read_text(encoding="utf-8"), environment)

    deck = environment["agent"](None)
    assert len(deck) == 60
    assert all(isinstance(card_id, int) for card_id in deck)
