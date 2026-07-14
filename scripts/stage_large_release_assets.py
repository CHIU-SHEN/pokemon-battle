"""Stage large generated datasets for a GitHub Release without duplicating disk usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    "data/processed/target_deck_profile.json",
    "data/reanalysis/README.md",
    "data/reanalysis/v1_labels_summary.json",
    "data/reanalysis/v1_candidates_summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stage_file(source: Path, destination: Path, prefer_hardlink: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"staged file has a different size: {destination}")
        return "existing"
    if prefer_hardlink:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release_assets/training_data_v1",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy large files instead of preferring same-volume hard links",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = []

    for relative in (*LARGE_FILES, *SUPPORT_FILES):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        method = stage_file(source, destination, prefer_hardlink=not args.copy)
        items.append({
            "path": relative,
            "role": "large_release_asset" if relative in LARGE_FILES else "support_metadata",
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "stage_method": method,
        })

    manifest = {
        "schema_version": "training_data_release_v1",
        "release_name": "pokemon-tcg-training-data-v1",
        "restore_root": "extract into the repository root",
        "large_file_count": len(LARGE_FILES),
        "total_large_bytes": sum(item["bytes"] for item in items if item["role"] == "large_release_asset"),
        "raw_replays_included": False,
        "raw_replays_note": "The 22.8 GB downloaded Kaggle raw replay cache is reproducible and redistribution-sensitive, so it is intentionally excluded.",
        "items": items,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Pokémon TCG Training Data v1\n\n"
        "This directory is staged for a GitHub Release and is ignored by normal Git history.\n\n"
        "Package the contents of this directory while preserving the `data/...` paths. "
        "Users restore the files by extracting the archive into the repository root.\n\n"
        "The five JSONL files are generated training artifacts. Check `MANIFEST.json` "
        "for byte sizes and SHA-256 hashes. The downloaded 22.8 GB raw replay cache is "
        "intentionally not included.\n\n"
        "Files staged as hard links do not consume a second full copy on this disk, but "
        "normal archive tools will still read and package their contents.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "large_files": len(LARGE_FILES),
        "total_large_bytes": manifest["total_large_bytes"],
        "total_large_gib": round(manifest["total_large_bytes"] / (1024 ** 3), 3),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
