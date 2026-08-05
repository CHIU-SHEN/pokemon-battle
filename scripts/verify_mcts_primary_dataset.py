#!/usr/bin/env python3
"""Verify a frozen MCTS primary dataset archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def _hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify(archive: Path) -> dict:
    archive = Path(archive).resolve()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    with archive.open("rb") as stream:
        if _hash(stream) != expected:
            raise ValueError("dataset archive checksum mismatch")
    prefix = "mcts-primary-dataset-v2/"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        if any(".." in PurePosixPath(name).parts or name.startswith("/") for name in members):
            raise ValueError("unsafe archive member")
        stream = bundle.extractfile(prefix + "MANIFEST.json")
        if stream is None:
            raise ValueError("dataset manifest missing")
        manifest = json.load(stream)
        if manifest.get("schema_version") != "mcts_primary_dataset_v2":
            raise ValueError("unsupported dataset schema")
        for relative, expected_hash in manifest["files"].items():
            stream = bundle.extractfile(prefix + relative)
            if stream is None or _hash(stream) != expected_hash:
                raise ValueError(f"dataset member hash mismatch: {relative}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps({"verified": True, **verify(args.archive)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
