#!/usr/bin/env python3
"""Build authority and Kaggle-budget V3 S16 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_NAME = "pokemon-tcg-v3-s16-authority"
KAGGLE_NAME = "pokemon-tcg-v3-s16-kaggle-60ms"
TEACHER_S128_NAME = "pokemon-tcg-v3-teacher-s128"
DIRECTORY_MAP = {"submission/agent": "agent", "submission/cg": "cg"}
FILE_MAP = {
    "candidates/v3_s16/main.py": "main.py",
    "data/cards.json": "data/cards.json",
    "data/card_tags.json": "data/card_tags.json",
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv": "data/high_score_decks/crustle_kangaskhan_cage/deck.csv",
    "src/arena/__init__.py": "src/arena/__init__.py",
    "src/arena/adapter_agent.py": "src/arena/adapter_agent.py",
    "src/rl/__init__.py": "src/rl/__init__.py",
    "src/rl/belief_puct_agent.py": "src/rl/belief_puct_agent.py",
    "src/rl/puct.py": "src/rl/puct.py",
    "src/rl/search_backend.py": "src/rl/search_backend.py",
    "src/rl/top2_ppo.py": "src/rl/top2_ppo.py",
    "src/rl/top2_rollout.py": "src/rl/top2_rollout.py",
    "src/train/__init__.py": "src/train/__init__.py",
    "src/train/adapter_model.py": "src/train/adapter_model.py",
    "src/train/features.py": "src/train/features.py",
    "src/train/shared_model.py": "src/train/shared_model.py",
}
FROZEN_MAP = {
    "artifacts/sl0_shared_full/best.pt": "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt": "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
    "artifacts/mcts_teacher_v3/primary-5k/train/best_safe_arena.pt": "model/best_safe_arena.pt",
}
VARIANTS = {
    "authority": {
        "package": AUTHORITY_NAME,
        "runtime": {"simulations": 16, "particles": 3, "max_depth": 10, "time_budget_seconds": 0.25, "game_budget_seconds": 120.0},
    },
    "kaggle-60ms": {
        "package": KAGGLE_NAME,
        "runtime": {"simulations": 16, "particles": 3, "max_depth": 10, "time_budget_seconds": 0.06, "game_budget_seconds": 5.0},
    },
    "teacher-s128": {
        "package": TEACHER_S128_NAME,
        "runtime": {"simulations": 128, "particles": 3, "max_depth": 10, "time_budget_seconds": 2.0, "game_budget_seconds": 120.0},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if (
            path.is_file()
            and path.name != ".DS_Store"
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        ):
            copy_file(path, destination / path.relative_to(source))


def _stage(root: Path, *, variant: str, code_root: Path, frozen_root: Path) -> dict:
    document = VARIANTS[variant]
    for source, destination in DIRECTORY_MAP.items():
        copy_directory(code_root / source, root / destination)
    for source, destination in FILE_MAP.items():
        copy_file(code_root / source, root / destination)
    copy_file(code_root / "data/high_score_decks/crustle_kangaskhan_cage/deck.csv", root / "deck.csv")
    missing = []
    for source, destination in FROZEN_MAP.items():
        path = frozen_root / source
        if not path.is_file():
            missing.append(source)
        else:
            copy_file(path, root / destination)
    (root / "runtime_config.json").write_text(json.dumps(document["runtime"], indent=2) + "\n", encoding="utf-8")
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": "v3_s16_candidate_v1",
        "variant": variant,
        "candidate_id": "crustle_kangaskhan_cage",
        "deck_id": "top2-primary-crustle-kangaskhan-cage-v1",
        "source_epoch": 44,
        "runtime_defaults": document["runtime"],
        "formal_submission_replacement_authorized": False,
        "kaggle_upload_ready": False,
        "missing_frozen_files": missing,
        "files": files,
    }
    (root / "CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _checksum(archive: Path) -> Path:
    path = archive.with_suffix(archive.suffix + ".sha256")
    path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return path


def build_all(output_dir: Path, *, code_root: Path = ROOT, frozen_root: Path = ROOT) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "_v3_s16_candidates"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    authority_root = work / AUTHORITY_NAME
    kaggle_root = work / KAGGLE_NAME
    teacher_s128_root = work / TEACHER_S128_NAME
    authority_manifest = _stage(authority_root, variant="authority", code_root=Path(code_root), frozen_root=Path(frozen_root))
    kaggle_manifest = _stage(kaggle_root, variant="kaggle-60ms", code_root=Path(code_root), frozen_root=Path(frozen_root))
    teacher_s128_manifest = _stage(teacher_s128_root, variant="teacher-s128", code_root=Path(code_root), frozen_root=Path(frozen_root))
    authority_archive = output_dir / f"{AUTHORITY_NAME}.tar.gz"
    kaggle_archive = output_dir / f"{KAGGLE_NAME}.tar.gz"
    teacher_s128_archive = output_dir / f"{TEACHER_S128_NAME}.tar.gz"
    with tarfile.open(authority_archive, "w:gz") as bundle:
        bundle.add(authority_root, arcname=AUTHORITY_NAME)
    with tarfile.open(kaggle_archive, "w:gz") as bundle:
        for path in sorted(kaggle_root.rglob("*")):
            bundle.add(path, arcname=path.relative_to(kaggle_root).as_posix(), recursive=False)
    with tarfile.open(teacher_s128_archive, "w:gz") as bundle:
        for path in sorted(teacher_s128_root.rglob("*")):
            bundle.add(path, arcname=path.relative_to(teacher_s128_root).as_posix(), recursive=False)
    result = {
        "authority_archive": authority_archive,
        "authority_checksum": _checksum(authority_archive),
        "authority_manifest": authority_manifest,
        "kaggle_archive": kaggle_archive,
        "kaggle_checksum": _checksum(kaggle_archive),
        "kaggle_manifest": kaggle_manifest,
        "teacher_s128_archive": teacher_s128_archive,
        "teacher_s128_checksum": _checksum(teacher_s128_archive),
        "teacher_s128_manifest": teacher_s128_manifest,
    }
    shutil.rmtree(work)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_submissions")
    args = parser.parse_args()
    result = build_all(args.output_dir, code_root=ROOT, frozen_root=ROOT)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, indent=2))
    manifests = (result["authority_manifest"], result["kaggle_manifest"], result["teacher_s128_manifest"])
    return 0 if all(not manifest["missing_frozen_files"] for manifest in manifests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
