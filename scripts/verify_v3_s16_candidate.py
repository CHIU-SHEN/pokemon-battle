#!/usr/bin/env python3
"""Verify an extracted V3 S16 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_RUNTIME = {
    "authority": {"simulations": 16, "particles": 3, "max_depth": 10, "time_budget_seconds": 0.25, "game_budget_seconds": 120.0},
    "kaggle-60ms": {"simulations": 16, "particles": 3, "max_depth": 10, "time_budget_seconds": 0.06, "game_budget_seconds": 5.0},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, *, expected_variant: str) -> dict:
    root = Path(root).resolve()
    manifest = json.loads((root / "CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["variant"] != expected_variant or manifest["runtime_defaults"] != EXPECTED_RUNTIME[expected_variant]:
        raise ValueError("candidate variant or runtime budget mismatch")
    if manifest["formal_submission_replacement_authorized"] or manifest["kaggle_upload_ready"]:
        raise ValueError("candidate must remain explicitly unpromoted")
    if manifest["missing_frozen_files"]:
        raise ValueError("candidate has missing frozen files")
    failures = [relative for relative, expected in manifest["files"].items() if not (root / relative).is_file() or sha256_file(root / relative) != expected]
    if failures:
        raise ValueError(f"candidate hash failures: {failures}")
    required = ("main.py", "deck.csv", "cg/api.py", "model/best_safe_arena.pt", "runtime_config.json")
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"candidate missing required files: {missing}")
    deck = [line for line in (root / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError("candidate deck must contain 60 cards")
    checkpoint = torch.load(root / "model/best_safe_arena.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "top2_mcts_checkpoint_v1" or checkpoint.get("source_epoch") != 44:
        raise ValueError("candidate checkpoint provenance mismatch")
    if checkpoint.get("candidate_id") != manifest["candidate_id"] or checkpoint.get("deck_id") != manifest["deck_id"]:
        raise ValueError("candidate checkpoint identity mismatch")
    return {"ok": True, "variant": expected_variant, "source_epoch": checkpoint["source_epoch"], "verified_files": len(manifest["files"]), "deck_cards": len(deck)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-variant", choices=tuple(EXPECTED_RUNTIME), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, expected_variant=args.expected_variant), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
