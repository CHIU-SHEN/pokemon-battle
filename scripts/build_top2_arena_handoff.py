#!/usr/bin/env python3
"""Build the self-contained Top2 Arena server handoff package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BASENAME = "pokemon-tcg-top2-arena-handoff-v2"
CANDIDATES = (
    "alakazam_battle_cage_split",
    "alakazam_neutralization_zone",
    "alakazam_nighttime_mine",
    "crustle_kangaskhan_cage",
    "crustle_kangaskhan_petrel",
    "cynthia_garchomp_roserade",
    "marnie_grimmsnarl_dudunsparce",
    "marnie_grimmsnarl_froslass",
    "marnie_grimmsnarl_tatsugiri",
    "mega_starmie_dusknoir",
)


REQUIRED_FILES = (
    "README.md",
    "项目进度.md",
    "TOP2_ARENA_SERVER_HANDOFF.md",
    "requirements-train.txt",
    "requirements-eval.txt",
    "scripts/build_top2_arena_handoff.py",
    "scripts/run_top10_adapter_smoke.py",
    "scripts/verify_top2_handoff.py",
    "eval/run_match.py",
    "eval/stats.py",
    "eval/compare_swapped.py",
    "eval/league.py",
    "src/arena/__init__.py",
    "src/arena/adapter_agent.py",
    "src/train/__init__.py",
    "src/train/shared_model.py",
    "src/train/adapter_model.py",
    "src/train/features.py",
    "src/train/eval_adapters.py",
    "submission/agent/__init__.py",
    "submission/agent/deck_profile_abomasnow.py",
    "submission/agent/fallback.py",
    "submission/agent/parser.py",
    "submission/agent/rules.py",
    "data/card_tags.json",
    "data/high_score_decks/top2_selection_policy.json",
    "artifacts/sl0_shared_full/best.pt",
    "reports/top10_adapter_offline_eval.json",
    "reports/top10_adapter_offline_eval.md",
    "reports/top10_adapter_online_smoke.json",
    "reports/top10_adapter_online_smoke.md",
    "reports/alakazam_battle_cage_split_retrain_eval.json",
    "reports/top10_adapter_v3_base_seed20260722_run1_review.md",
    "tests/README.md",
    "tests/test_adapter_arena_agent.py",
    "tests/fixtures/observations.json",
)


OPTIONAL_FILES = (
    "artifacts/sl0_shared_full/run_config.json",
    "reports/top10_adapter_v3_base_seed20260722_run1_eval.json",
    "docs/cleanup-reports/REPO_CLEANUP_2026-07-29.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative: str, package_root: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = package_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(relative: str, package_root: Path) -> None:
    source = ROOT / relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = package_root / relative
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def build(output_dir: Path) -> tuple[Path, Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}_build"
    package_root = staging / PACKAGE_BASENAME
    if staging.exists():
        shutil.rmtree(staging)
    package_root.mkdir(parents=True)

    for relative in REQUIRED_FILES:
        copy_file(relative, package_root)
    for relative in OPTIONAL_FILES:
        if (ROOT / relative).is_file():
            copy_file(relative, package_root)
    copy_tree("submission/cg", package_root)

    for candidate in CANDIDATES:
        copy_file(f"data/high_score_decks/{candidate}/deck.csv", package_root)
        copy_file(f"artifacts/adapters_top10/{candidate}/best.pt", package_root)
        copy_file(f"artifacts/adapters_top10/{candidate}/metrics.json", package_root)

    payload_files = sorted(
        path for path in package_root.rglob("*") if path.is_file()
    )
    hashes = {
        path.relative_to(package_root).as_posix(): sha256(path)
        for path in payload_files
    }
    manifest = {
        "schema_version": "top2_arena_handoff_manifest_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package": PACKAGE_BASENAME,
        "purpose": "Top10 online Arena, external matrix, and Top2 freeze; no retraining",
        "candidates": list(CANDIDATES),
        "payload_file_count": len(hashes),
        "payload_bytes": sum(path.stat().st_size for path in payload_files),
        "excluded": [
            "data/training/training_decisions_v1.jsonl",
            "raw Arena games",
            "last.pt checkpoints",
            "server result archives",
        ],
        "sha256": hashes,
    }
    (package_root / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_root / "HANDOFF_SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in hashes.items()),
        encoding="utf-8",
    )

    archive = output_dir / f"{PACKAGE_BASENAME}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(package_root, arcname=PACKAGE_BASENAME)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    shutil.rmtree(staging)
    return archive, sidecar, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "server_uploads")
    args = parser.parse_args()
    archive, sidecar, manifest = build(args.output_dir.resolve())
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256_file": str(sidecar),
                "archive_sha256": sha256(archive),
                "payload_file_count": manifest["payload_file_count"],
                "payload_bytes": manifest["payload_bytes"],
                "candidates": len(manifest["candidates"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
