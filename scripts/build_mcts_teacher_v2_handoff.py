#!/usr/bin/env python3
"""Build the checksum-verified primary MCTS teacher v2 server handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BASENAME = "mcts-distill-v2-teacher"
AUTHORITATIVE_ARCHIVE = "artifacts/top2-mcts-complete-results-20260804.tar.gz"
AUTHORITATIVE_SHA256 = "f926fbe822d18321d3e083bd30fd60a73da6f35517327f69d0e7bd44262cb531"
CODE_DIRECTORIES = ("src", "eval", "submission/agent", "submission/cg")
CODE_FILES = (
    "config/top2_rl_policy.json",
    "requirements-train.txt",
    "requirements-eval.txt",
    "scripts/collect_top2_mcts.py",
    "scripts/evaluate_top2_mcts.py",
    "scripts/train_top2_mcts.py",
    "scripts/run_mcts_teacher_smoke.py",
    "scripts/verify_mcts_teacher_v2_handoff.py",
    "jobs/mcts_teacher_v2_resilient.sh",
    "docs/operations/MCTS_TEACHER_V2_SERVER_HANDOFF.md",
    "data/cards.json",
    "data/card_tags.json",
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv",
)
FROZEN_FILES = (
    "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
    AUTHORITATIVE_ARCHIVE,
    AUTHORITATIVE_ARCHIVE + ".sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source_root: Path, destination_root: Path, relative: str) -> list[str]:
    source = source_root / relative
    if source.is_file():
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return [relative]
    if not source.is_dir():
        return []
    copied = []
    for path in source.rglob("*"):
        if (
            not path.is_file()
            or path.suffix == ".pyc"
            or path.name == ".DS_Store"
            or "__pycache__" in path.parts
        ):
            continue
        child = path.relative_to(source_root).as_posix()
        destination = destination_root / child
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(child)
    return copied


def build(output_dir: Path, *, code_root: Path = ROOT, frozen_root: Path = ROOT):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}"
    if staging.exists():
        shutil.rmtree(staging)
    package_root = staging / PACKAGE_BASENAME
    package_root.mkdir(parents=True)
    files = []
    for relative in (*CODE_DIRECTORIES, *CODE_FILES):
        files.extend(_copy(Path(code_root), package_root, relative))
    missing = []
    for relative in FROZEN_FILES:
        copied = _copy(Path(frozen_root), package_root, relative)
        files.extend(copied)
        if not copied:
            missing.append(relative)
    authoritative = Path(frozen_root) / AUTHORITATIVE_ARCHIVE
    if authoritative.is_file() and sha256_file(authoritative) != AUTHORITATIVE_SHA256:
        raise ValueError("authoritative MCTS archive hash mismatch")
    manifest = {
        "schema_version": "mcts_teacher_v2_handoff_v1",
        "branch": "primary",
        "authoritative_archive": AUTHORITATIVE_ARCHIVE,
        "authoritative_archive_sha256": AUTHORITATIVE_SHA256,
        "smoke_max_wall_seconds": 1800,
        "train_max_wall_seconds": 21600,
        "iteration_max_wall_seconds": 86400,
        "submission_replacement_authorized": False,
        "arena_automatic_start": False,
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
    archive, checksum, manifest = build(
        args.output_dir,
        code_root=ROOT,
        frozen_root=args.frozen_root.resolve(),
    )
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}, ensure_ascii=False))
    return 0 if not manifest["missing_frozen_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
