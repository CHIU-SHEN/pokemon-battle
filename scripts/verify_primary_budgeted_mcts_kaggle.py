#!/usr/bin/env python3
"""Verify an extracted primary budgeted-MCTS Kaggle submission."""

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
    manifest = json.loads((root / "KAGGLE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["candidate_id"] != "crustle_kangaskhan_cage":
        raise ValueError("unexpected candidate identity")
    if not manifest["kaggle_upload_ready"]:
        raise ValueError("manifest is not marked Kaggle-upload-ready")
    if manifest["formal_submission_replacement_authorized"]:
        raise ValueError("package must not authorize formal replacement")

    required = ("main.py", "deck.csv", "cg/api.py")
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"missing top-level Kaggle files: {missing}")

    failures = [
        relative
        for relative, expected in manifest["files"].items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected
    ]
    if failures:
        raise ValueError(f"hash failures: {failures}")

    deck = [
        int(line)
        for line in (root / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"top-level deck must contain 60 cards, found {len(deck)}")
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if "petrel" in path.as_posix().lower() or "reserve" in path.as_posix().lower()
    ]
    if forbidden:
        raise ValueError(f"reserve assets found: {forbidden}")
    return {
        "ok": True,
        "candidate_id": manifest["candidate_id"],
        "deck_cards": len(deck),
        "verified_files": len(manifest["files"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
