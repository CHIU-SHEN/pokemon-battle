#!/usr/bin/env python3
"""Freeze an audited MCTS collection into a checksum-verified archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.mcts_collection import audit_collection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, output: Path, *, identity: dict[str, str]):
    source, output = Path(source).resolve(), Path(output).resolve()
    report = audit_collection(source, identity, require_all_splits=True)
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "_mcts-primary-dataset-v2"
    if stage.exists():
        shutil.rmtree(stage)
    games_root = stage / "games"
    games_root.mkdir(parents=True)
    hashes = {}
    for index, path in enumerate(sorted(set(source.glob("**/games/game*.json")))):
        relative = f"games/game_{index:06d}.json"
        shutil.copy2(path, stage / relative)
        hashes[relative] = sha256_file(stage / relative)
    manifest = {
        "schema_version": "mcts_primary_dataset_v2",
        "identity": identity,
        "totals": {key: report[key] for key in ("games", "samples", "nodes", "exceptions", "illegal_actions", "fallbacks", "fallback_rate", "splits")},
        "game_ids": report["game_ids"],
        "files": hashes,
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = output / "mcts-primary-dataset-v2.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(stage, arcname="mcts-primary-dataset-v2")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    shutil.rmtree(stage)
    return archive, checksum, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    args = parser.parse_args()
    archive, checksum, manifest = build(args.source, args.output, identity={"branch": "primary", "deck_id": args.deck_id})
    print(json.dumps({"archive": str(archive), "checksum": str(checksum), **manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
