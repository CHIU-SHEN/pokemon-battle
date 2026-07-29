#!/usr/bin/env python3
"""Build the self-contained Top2 reinforcement-learning handoff package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_BASENAME = "pokemon-tcg-top2-rl-handoff-v2"

REQUIRED_FILES = (
    "README.md",
    "项目进度.md",
    "TOP2_RL_SERVER_HANDOFF.md",
    "requirements-eval.txt",
    "requirements-train.txt",
    "config/top2_rl_policy.json",
    "config/top2_rl_selected.json",
    "data/card_tags.json",
    "data/high_score_decks/top2_selection_policy.json",
    "data/high_score_decks/crustle_kangaskhan_cage/deck.csv",
    "data/high_score_decks/crustle_kangaskhan_petrel/deck.csv",
    "reports/top2_freeze_report.json",
    "reports/top2_freeze_report.md",
    "reports/top2_local_pilot_report.json",
    "reports/top2_local_pilot_report.md",
    "artifacts/sl0_shared_full/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt",
    "artifacts/adapters_top10/crustle_kangaskhan_petrel/best.pt",
    "eval/run_match.py",
    "eval/stats.py",
    "src/arena/__init__.py",
    "src/arena/adapter_agent.py",
    "src/arena/ppo_agent.py",
    "src/rl/__init__.py",
    "src/rl/top2_rollout.py",
    "src/rl/top2_ppo.py",
    "src/rl/pilot.py",
    "src/train/__init__.py",
    "src/train/adapter_model.py",
    "src/train/features.py",
    "src/train/shared_data.py",
    "src/train/shared_model.py",
    "scripts/build_top2_rl_handoff.py",
    "scripts/verify_top2_rl_handoff.py",
    "scripts/collect_top2_rollouts.py",
    "scripts/train_top2_ppo.py",
    "scripts/evaluate_top2_ppo.py",
    "scripts/evaluate_top2_ppo_holdout.py",
    "scripts/run_top2_local_pilot.py",
    "scripts/select_top2_v1_candidates.py",
    "scripts/run_top2_v1_reanalysis.py",
    "scripts/convert_bad_cases.py",
    "scripts/select_v1_candidates.py",
    "scripts/run_v1_reanalysis.py",
    "src/train/observed_schema.py",
    "tests/test_top2_rl_handoff.py",
    "tests/test_top2_rl_package_smoke.py",
    "tests/fixtures/observations.json",
    "docs/superpowers/specs/2026-07-29-top2-rl-handoff-design.md",
    "docs/superpowers/specs/2026-07-29-top2-local-pilot-design.md",
)

REQUIRED_TREES = ("submission/cg", "submission/agent")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_handoff_archive(package_root: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    """Add a payload manifest and archive one already-populated package root."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_files = sorted(
        path for path in package_root.rglob("*")
        if path.is_file() and path.name not in {"HANDOFF_MANIFEST.json", "HANDOFF_SHA256SUMS"}
    )
    hashes = {path.relative_to(package_root).as_posix(): sha256(path) for path in payload_files}
    manifest = {
        "schema_version": "top2_rl_handoff_manifest_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package": PACKAGE_BASENAME,
        "purpose": "Top2 server RL handoff with local RTX 5060 pilot evidence and preliminary hyperparameters",
        "payload_file_count": len(hashes),
        "payload_bytes": sum(path.stat().st_size for path in payload_files),
        "excluded": [
            "full supervised training JSONL",
            "raw rollout output",
            "local pilot trial checkpoints",
            "valid/test holdout training access",
            "last.pt historical checkpoints",
            "submission/deck.csv replacement",
        ],
        "sha256": hashes,
    }
    (package_root / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package_root / "HANDOFF_SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in hashes.items()), encoding="utf-8"
    )
    archive = output_dir / f"{PACKAGE_BASENAME}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(package_root, arcname=PACKAGE_BASENAME)
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, sidecar, manifest


def resolve_source(relative: str, code_root: Path, frozen_source_root: Path) -> Path:
    """Prefer edited code, but always source frozen model artifacts from the frozen root."""

    if relative.startswith("artifacts/"):
        return frozen_source_root / relative
    edited = code_root / relative
    return edited if edited.exists() else frozen_source_root / relative


def build(output_dir: Path, frozen_source_root: Path, code_root: Path = ROOT) -> tuple[Path, Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"_{PACKAGE_BASENAME}_build"
    if staging.exists():
        shutil.rmtree(staging)
    package_root = staging / PACKAGE_BASENAME
    package_root.mkdir(parents=True)
    for relative in REQUIRED_FILES:
        source = resolve_source(relative, code_root, frozen_source_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in REQUIRED_TREES:
        source = resolve_source(relative, code_root, frozen_source_root)
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(
            source,
            package_root / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "model.npz"),
        )
    archive, sidecar, manifest = write_handoff_archive(package_root, output_dir)
    shutil.rmtree(staging)
    return archive, sidecar, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "server_uploads")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Root containing frozen checkpoints.")
    args = parser.parse_args()
    archive, sidecar, manifest = build(args.output_dir.resolve(), args.source_root.resolve())
    print(json.dumps({
        "archive": str(archive),
        "sha256_file": str(sidecar),
        "archive_sha256": sha256(archive),
        "payload_file_count": manifest["payload_file_count"],
        "payload_bytes": manifest["payload_bytes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
