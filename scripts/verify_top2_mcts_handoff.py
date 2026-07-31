#!/usr/bin/env python3
"""Verify an extracted Top2 MCTS pilot handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    manifest = json.loads((root / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["branches"] != ["primary", "reserve"]:
        raise ValueError("unexpected branches")
    if manifest["submission_replacement_authorized"]:
        raise ValueError("pilot must not authorize submission replacement")
    failures = [
        relative
        for relative, expected in manifest["files"].items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected
    ]
    if failures:
        raise ValueError(f"handoff hash failures: {failures}")
    return {"ok": True, "verified_files": len(manifest["files"]), "branches": manifest["branches"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
