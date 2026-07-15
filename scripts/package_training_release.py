"""Create a compressed, GitHub-Release-ready training data package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

LARGE_FILES = (
    "data/training/training_decisions_v1.jsonl",
    "data/processed/kaggle_decisions.jsonl",
    "data/processed/bad_case_decisions.jsonl",
    "data/reanalysis/v1_labels.jsonl",
    "data/reanalysis/v1_candidates.jsonl",
)

SUPPORT_FILES = (
    "data/training/README.md",
    "data/training/training_manifest_v1.json",
    "data/processed/README.md",
    "data/processed/kaggle_conversion_summary.json",
    "data/processed/bad_case_conversion_summary.json",
    "data/processed/local_data_audit.json",
    "data/processed/target_deck_profile.json",
    "data/reanalysis/README.md",
    "data/reanalysis/v1_labels_summary.json",
    "data/reanalysis/v1_candidates_summary.json",
    "data/cards.json",
    "data/card_tags.json",
    "data/card_tag_full_audit.json",
    "data/manual_overrides.json",
    "data/external/acquisition_manifest.json",
    "data/external/kaggle_replays/replay_index.json",
)

GITHUB_ASSET_LIMIT = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def gzip_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release_assets/training_data_v2")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = []

    for relative in LARGE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / f"{relative}.gz"
        gzip_file(source, destination)
        if destination.stat().st_size >= GITHUB_ASSET_LIMIT:
            raise RuntimeError(f"compressed asset still exceeds GitHub's 2 GiB limit: {destination}")
        items.append({
            "path": relative,
            "asset": destination.relative_to(output).as_posix(),
            "compression": "gzip",
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "asset_bytes": destination.stat().st_size,
            "asset_sha256": sha256(destination),
        })

    support_items = []
    for relative in SUPPORT_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        support_items.append({
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })

    manifest = {
        "schema_version": "training_data_release_v2",
        "release_name": "pokemon-tcg-training-data-v2",
        "ruleset": "ptcg_abc_2026_simulation_designated_pool_v1",
        "restore_root": "repository root",
        "raw_replays_included": False,
        "raw_replays_note": "The 22.8 GiB authenticated Kaggle replay cache and obsolete .part file are intentionally excluded.",
        "large_files": items,
        "support_files": support_items,
        "total_original_bytes": sum(item["bytes"] for item in items),
        "total_asset_bytes": sum(item["asset_bytes"] for item in items),
        "github_asset_limit_bytes": GITHUB_ASSET_LIMIT,
        "ok": all(item["asset_bytes"] < GITHUB_ASSET_LIMIT for item in items),
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Pokémon TCG Training Data v2\n\n"
        "This package contains the final derived training artifacts for ruleset "
        "`ptcg_abc_2026_simulation_designated_pool_v1`.\n\n"
        "Upload the five `.jsonl.gz` files, `MANIFEST.json`, and optionally the support metadata tree "
        "to the same GitHub Release. Restore a dataset with `gzip -dk <file>.jsonl.gz`, preserving its "
        "`data/...` path. Verify both compressed and restored SHA-256 values against `MANIFEST.json`.\n\n"
        "Authenticated raw Kaggle replays and `kaggle_decisions.jsonl.part` are deliberately excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "assets": len(items),
        "original_gib": round(manifest["total_original_bytes"] / 1024**3, 3),
        "compressed_gib": round(manifest["total_asset_bytes"] / 1024**3, 3),
        "ok": manifest["ok"],
    }, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
