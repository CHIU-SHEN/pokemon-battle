#!/usr/bin/env python3
"""Build the primary budgeted-MCTS agent as a Kaggle-ready tar.gz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_primary_budgeted_mcts_candidate import (
    DIRECTORY_MAP,
    FILE_MAP as CANDIDATE_FILE_MAP,
    FROZEN_FILES,
    copy_directory,
    copy_file,
    sha256_file,
)


PACKAGE_BASENAME = "primary_budgeted_mcts_v1"
FILE_MAP = {
    **{
        source: destination
        for source, destination in CANDIDATE_FILE_MAP.items()
        if destination not in {"README.md", "verify_candidate.py"}
    },
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv": "deck.csv",
    "candidates/primary_budgeted_mcts/README.md": "README.md",
    "scripts/verify_primary_budgeted_mcts_kaggle.py": "verify_submission.py",
}


def build(output_dir: Path, *, code_root: Path = ROOT, frozen_root: Path = ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for source, destination in DIRECTORY_MAP.items():
        copy_directory(code_root / source, staging / destination)
    for source, destination in FILE_MAP.items():
        copy_file(code_root / source, staging / destination)

    missing = []
    for relative in FROZEN_FILES:
        source = frozen_root / relative
        if not source.is_file():
            missing.append(relative)
        else:
            copy_file(source, staging / relative)

    files = {
        path.relative_to(staging).as_posix(): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "primary_budgeted_mcts_kaggle_v1",
        "candidate_id": "crustle_kangaskhan_cage",
        "deck_id": "top2-primary-crustle-kangaskhan-cage-v1",
        "kaggle_upload_ready": not missing,
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
    (staging / "KAGGLE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    archive = output_dir / f"{PACKAGE_BASENAME}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(staging.rglob("*")):
            bundle.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    shutil.rmtree(staging)
    return archive, checksum, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_submissions")
    parser.add_argument("--frozen-root", type=Path, default=ROOT)
    args = parser.parse_args()
    archive, checksum, manifest = build(
        args.output_dir.resolve(), code_root=ROOT, frozen_root=args.frozen_root.resolve()
    )
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}, indent=2))
    return 0 if manifest["kaggle_upload_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
