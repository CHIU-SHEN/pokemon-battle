#!/usr/bin/env python3
"""Verify the unified MCTS teacher handoff."""

from __future__ import annotations
import argparse, hashlib, json, tarfile
from pathlib import Path, PurePosixPath

def _hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def verify(archive: Path) -> dict:
    archive = Path(archive).resolve()
    expected = archive.with_suffix(archive.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    with archive.open("rb") as stream:
        if _hash(stream) != expected: raise ValueError("handoff archive checksum mismatch")
    prefix = "mcts-teacher-v2-all-in-one/"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {item.name: item for item in bundle.getmembers() if item.isfile()}
        if any(".." in PurePosixPath(name).parts for name in members): raise ValueError("unsafe member")
        stream = bundle.extractfile(prefix + "HANDOFF_MANIFEST.json")
        if stream is None: raise ValueError("manifest missing")
        manifest = json.load(stream)
        if manifest.get("schema_version") != "mcts_teacher_v2_all_in_one_v1": raise ValueError("unsupported manifest")
        for relative, expected_hash in manifest["files"].items():
            stream = bundle.extractfile(prefix + relative)
            if stream is None or _hash(stream) != expected_hash: raise ValueError(f"member hash mismatch: {relative}")
            if relative.endswith(".sh"):
                stream = bundle.extractfile(prefix + relative)
                if b"\r" in stream.read(): raise ValueError(f"CRLF shell member: {relative}")
    return manifest

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("archive", type=Path); args = parser.parse_args()
    print(json.dumps({"verified": True, "schema_version": verify(args.archive)["schema_version"]})); return 0

if __name__ == "__main__": raise SystemExit(main())
