#!/usr/bin/env python3
"""Verify archive, manifest, member hashes, and authoritative MCTS input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


EXPECTED_SCHEMA = "mcts_teacher_v2_handoff_v1"


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify(archive: Path) -> dict:
    archive = Path(archive).resolve()
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    expected_archive_hash = checksum_path.read_text(encoding="utf-8").split()[0].lower()
    with archive.open("rb") as stream:
        actual_archive_hash = _sha256_stream(stream)
    if actual_archive_hash != expected_archive_hash:
        raise ValueError("handoff archive checksum mismatch")
    prefix = "mcts-distill-v2-teacher/"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        if any(".." in PurePosixPath(name).parts or name.startswith("/") for name in members):
            raise ValueError("unsafe archive member path")
        manifest_stream = bundle.extractfile(prefix + "HANDOFF_MANIFEST.json")
        if manifest_stream is None:
            raise ValueError("handoff manifest missing")
        manifest = json.load(manifest_stream)
        if manifest.get("schema_version") != EXPECTED_SCHEMA:
            raise ValueError("unsupported handoff manifest")
        for relative, expected in manifest["files"].items():
            stream = bundle.extractfile(prefix + relative)
            if stream is None or _sha256_stream(stream) != expected:
                raise ValueError(f"handoff member hash mismatch: {relative}")
        authoritative = manifest["authoritative_archive"]
        stream = bundle.extractfile(prefix + authoritative)
        if stream is None or _sha256_stream(stream) != manifest["authoritative_archive_sha256"]:
            raise ValueError("authoritative MCTS archive hash mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    manifest = verify(args.archive)
    print(json.dumps({"verified": True, "schema_version": manifest["schema_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
