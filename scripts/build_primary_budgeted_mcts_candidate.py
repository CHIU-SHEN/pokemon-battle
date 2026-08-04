#!/usr/bin/env python3
"""Build an isolated checksum-verified primary budgeted-MCTS candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BASENAME = "pokemon-tcg-primary-budgeted-mcts-v1"
DIRECTORY_MAP = {
    "submission/agent": "agent",
    "submission/cg": "cg",
}
FILE_MAP = {
    "candidates/primary_budgeted_mcts/main.py": "main.py",
    "candidates/primary_budgeted_mcts/README.md": "README.md",
    "scripts/verify_primary_budgeted_mcts_candidate.py": "verify_candidate.py",
    "requirements-eval.txt": "requirements-eval.txt",
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
FROZEN_FILES = (
    "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
)


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
            not path.is_file()
            or path.name == ".DS_Store"
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".jsonl"}
        ):
            continue
        copy_file(path, destination / path.relative_to(source))


def build(output_dir: Path, *, code_root: Path = ROOT, frozen_root: Path = ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}"
    if staging.exists():
        shutil.rmtree(staging)
    package_root = staging / PACKAGE_BASENAME
    package_root.mkdir(parents=True)
    for source, destination in DIRECTORY_MAP.items():
        copy_directory(code_root / source, package_root / destination)
    for source, destination in FILE_MAP.items():
        copy_file(code_root / source, package_root / destination)
    missing = []
    for relative in FROZEN_FILES:
        source = frozen_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        copy_file(source, package_root / relative)
    files = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "primary_budgeted_mcts_candidate_v1",
        "candidate_id": "crustle_kangaskhan_cage",
        "deck_id": "top2-primary-crustle-kangaskhan-cage-v1",
        "formal_submission_replacement_authorized": False,
        "runtime_defaults": {
            "simulations": 8,
            "particles": 1,
            "max_depth": 4,
            "time_budget_seconds": 0.030,
            "game_budget_seconds": 2.0,
        },
        "missing_frozen_files": missing,
        "files": files,
    }
    (package_root / "CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    archive = output_dir / f"{PACKAGE_BASENAME}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(package_root, arcname=PACKAGE_BASENAME)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    shutil.rmtree(staging)
    return archive, checksum, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "server_uploads")
    parser.add_argument("--frozen-root", type=Path, default=ROOT)
    args = parser.parse_args()
    archive, checksum, manifest = build(
        args.output_dir.resolve(), code_root=ROOT, frozen_root=args.frozen_root.resolve()
    )
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}, indent=2))
    return 0 if not manifest["missing_frozen_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
