#!/usr/bin/env python3
"""Verify the MCTS teacher v3 quality-gated handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


PACKAGE = "mcts-teacher-v3-quality-gated"
SCHEMA = "mcts_teacher_v3_quality_gated_v1"


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
            raise ValueError("handoff archive checksum mismatch")
    prefix = f"{PACKAGE}/"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {item.name: item for item in bundle.getmembers() if item.isfile()}
        if any(".." in PurePosixPath(name).parts for name in members):
            raise ValueError("unsafe archive member")
        stream = bundle.extractfile(prefix + "HANDOFF_MANIFEST.json")
        if stream is None:
            raise ValueError("handoff manifest missing")
        manifest = json.load(stream)
        if manifest.get("schema_version") != SCHEMA:
            raise ValueError("unsupported handoff manifest")
        if manifest.get("missing_frozen_files"):
            raise ValueError("handoff has missing frozen files")
        for relative, expected_hash in manifest.get("files", {}).items():
            member_name = prefix + relative
            stream = bundle.extractfile(member_name)
            if stream is None or _hash(stream) != expected_hash:
                raise ValueError(f"handoff member hash mismatch: {relative}")
            if relative.endswith(".sh"):
                shell_stream = bundle.extractfile(member_name)
                if shell_stream is None or b"\r" in shell_stream.read():
                    raise ValueError(f"CRLF shell member: {relative}")
    return {"verified": True, **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = verify(args.archive)
    print(json.dumps({"verified": True, "schema_version": result["schema_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
