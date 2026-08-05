#!/usr/bin/env python3
"""Build the quality-gated MCTS teacher v3 server handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


PACKAGE = "mcts-teacher-v3-quality-gated"
AUTHORITATIVE = "artifacts/top2-mcts-complete-results-20260804.tar.gz"
AUTHORITATIVE_SHA256 = "f926fbe822d18321d3e083bd30fd60a73da6f35517327f69d0e7bd44262cb531"
DIRECTORIES = ("src", "eval", "submission/agent", "submission/cg")
FILES = (
    "config/top2_rl_policy.json",
    "requirements-train.txt",
    "requirements-eval.txt",
    "scripts/evaluate_top2_mcts.py",
    "scripts/gate_mcts_teacher.py",
    "scripts/collect_top2_mcts.py",
    "scripts/audit_mcts_collection.py",
    "scripts/build_mcts_primary_dataset.py",
    "scripts/verify_mcts_primary_dataset.py",
    "scripts/train_top2_mcts.py",
    "scripts/verify_mcts_teacher_v3_handoff.py",
    "jobs/mcts_teacher_v3_quality_gated.sh",
    "docs/MCTS_TEACHER_V3_SERVER_RUNBOOK.md",
    "docs/MCTS_HYBRID_POLICY_DECISION.md",
    "data/cards.json",
    "data/card_tags.json",
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv",
)
FROZEN = (
    "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
    AUTHORITATIVE,
    AUTHORITATIVE + ".sha256",
)
DEFAULTS = {
    "teacher_gate_games": 400,
    "teacher_min_win_rate": 0.58,
    "target_games": 5000,
    "simulations": 128,
    "particles": 3,
    "max_depth": 10,
    "workers": 16,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(root: Path, destination: Path, relative: str) -> list[str]:
    source = root / relative
    paths = [source] if source.is_file() else list(source.rglob("*")) if source.is_dir() else []
    copied: list[str] = []
    for path in paths:
        if (
            not path.is_file()
            or path.name == ".DS_Store"
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
        ):
            continue
        child = path.relative_to(root).as_posix()
        target = destination / child
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if path.suffix == ".sh":
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        target.write_bytes(data)
        copied.append(child)
    return copied


def build(output: Path, *, code_root: Path, frozen_root: Path):
    output = output.resolve()
    code_root = code_root.resolve()
    frozen_root = frozen_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage = output / f"_{PACKAGE}"
    if stage.exists():
        shutil.rmtree(stage)
    package_root = stage / PACKAGE
    package_root.mkdir(parents=True)
    members: list[str] = []
    for relative in (*DIRECTORIES, *FILES):
        members.extend(_copy(code_root, package_root, relative))
    missing: list[str] = []
    for relative in FROZEN:
        copied = _copy(frozen_root, package_root, relative)
        members.extend(copied)
        if not copied:
            missing.append(relative)
    authoritative = frozen_root / AUTHORITATIVE
    if authoritative.is_file() and sha256_file(authoritative) != AUTHORITATIVE_SHA256:
        raise ValueError("authoritative archive hash mismatch")
    manifest = {
        "schema_version": "mcts_teacher_v3_quality_gated_v1",
        "branch": "primary",
        "defaults": DEFAULTS,
        "authoritative_archive_sha256": AUTHORITATIVE_SHA256,
        "missing_frozen_files": missing,
        "files": {},
    }
    for relative in sorted(set(members)):
        manifest["files"][relative] = sha256_file(package_root / relative)
    (package_root / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    archive = output / f"{PACKAGE}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(package_root, arcname=PACKAGE)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    shutil.rmtree(stage)
    return archive, checksum, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("server_uploads"))
    parser.add_argument("--frozen-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive, checksum, manifest = build(
        args.output_dir, code_root=root, frozen_root=args.frozen_root
    )
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}))
    return 0 if not manifest["missing_frozen_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
