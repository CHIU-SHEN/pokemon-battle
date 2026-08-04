#!/usr/bin/env python3
"""Verify a primary budgeted-MCTS candidate directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    manifest = json.loads((root / "CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["formal_submission_replacement_authorized"]:
        raise ValueError("candidate must not authorize formal replacement")
    if manifest["candidate_id"] != "crustle_kangaskhan_cage":
        raise ValueError("unexpected candidate identity")
    failures = [
        relative
        for relative, expected in manifest["files"].items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected
    ]
    if failures:
        raise ValueError(f"candidate hash failures: {failures}")
    forbidden = [path for path in root.rglob("*") if "petrel" in path.as_posix().lower()]
    if forbidden:
        raise ValueError("reserve assets found in primary candidate")
    return {"ok": True, "verified_files": len(manifest["files"]), "candidate_id": manifest["candidate_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
