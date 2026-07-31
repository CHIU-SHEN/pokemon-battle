#!/usr/bin/env python3
"""Build a checksum-verified Top2 MCTS pilot handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BASENAME = "pokemon-tcg-top2-mcts-pilot-v1"
SOURCE_DIRECTORIES = ("src", "eval", "submission/agent", "submission/cg")
SOURCE_FILES = (
    "config/top2_rl_policy.json",
    "requirements-train.txt",
    "requirements-eval.txt",
    "TOP2_MCTS_SERVER_HANDOFF.md",
    "scripts/collect_top2_mcts.py",
    "scripts/train_top2_mcts.py",
    "scripts/evaluate_top2_mcts.py",
    "scripts/run_top2_mcts_pilot.py",
    "scripts/build_top2_mcts_handoff.py",
    "scripts/verify_top2_mcts_handoff.py",
    "jobs/top2_mcts_pilot_single_node.sh",
    "data/cards.json",
    "data/card_tags.json",
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv",
    "data/high_score_decks/crustle_kangaskhan_petrel/deck.csv",
)
FROZEN_FILES = (
    "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_petrel/best.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_relative(source_root: Path, destination_root: Path, relative: str) -> list[str]:
    source = source_root / relative
    copied = []
    if source.is_file():
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return [relative]
    if source.is_dir():
        for path in source.rglob("*"):
            if (
                path.is_file()
                and path.suffix not in {".pyc", ".jsonl"}
                and path.name != ".DS_Store"
                and "__pycache__" not in path.parts
            ):
                child = path.relative_to(source_root).as_posix()
                destination = destination_root / child
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                copied.append(child)
    return copied


def build(output_dir: Path, code_root: Path = ROOT, frozen_root: Path = ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}"
    if staging.exists():
        shutil.rmtree(staging)
    package_root = staging / PACKAGE_BASENAME
    package_root.mkdir(parents=True)
    files = []
    for relative in SOURCE_DIRECTORIES + SOURCE_FILES:
        files.extend(copy_relative(code_root, package_root, relative))
    missing = []
    for relative in FROZEN_FILES:
        copied = copy_relative(frozen_root, package_root, relative)
        files.extend(copied)
        if not copied:
            missing.append(relative)
    manifest = {
        "schema_version": "top2_mcts_handoff_v1",
        "branches": ["primary", "reserve"],
        "smoke_games_per_branch": 10,
        "pilot_games_per_branch": 200,
        "simulations": 32,
        "particles": 3,
        "max_depth": 8,
        "submission_replacement_authorized": False,
        "missing_frozen_files": missing,
        "files": {},
    }
    for relative in sorted(set(files)):
        manifest["files"][relative] = sha256_file(package_root / relative)
    (package_root / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
    archive, checksum, manifest = build(args.output_dir.resolve(), ROOT, args.frozen_root.resolve())
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}, ensure_ascii=False, indent=2))
    return 0 if not manifest["missing_frozen_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
