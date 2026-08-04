from __future__ import annotations

from pathlib import Path
import tarfile


def test_package_contains_only_primary_runtime_and_frozen_assets(tmp_path: Path) -> None:
    from scripts.build_primary_budgeted_mcts_candidate import PACKAGE_BASENAME, build

    code_root = Path(__file__).resolve().parents[1]
    frozen_root = code_root.parents[1]
    formal_main = frozen_root / "submission/main.py"
    before = formal_main.read_bytes()

    archive, checksum, manifest = build(tmp_path, code_root=code_root, frozen_root=frozen_root)

    assert archive.name == f"{PACKAGE_BASENAME}.tar.gz"
    assert archive.is_file() and checksum.is_file()
    assert manifest["formal_submission_replacement_authorized"] is False
    assert manifest["candidate_id"] == "crustle_kangaskhan_cage"
    assert manifest["runtime_defaults"]["time_budget_seconds"] == 0.030
    assert not manifest["missing_frozen_files"]
    assert formal_main.read_bytes() == before

    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
    assert f"{PACKAGE_BASENAME}/main.py" in names
    assert any(name.endswith("crustle_kangaskhan_cage/deck.csv") for name in names)
    assert not any("petrel" in name or "reserve" in name for name in names)
    assert not any("samples/games" in name or name.endswith("top2_mcts_checkpoint_v1") for name in names)
    assert not any(name.endswith(".DS_Store") for name in names)
    assert not any("/src/train/train_" in name or "/src/rl/mcts_train.py" in name for name in names)
