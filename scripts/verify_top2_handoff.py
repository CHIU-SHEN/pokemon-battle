#!/usr/bin/env python3
"""Verify a self-contained Top2 Arena server handoff."""

from __future__ import annotations

import csv
import json
import pathlib
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "alakazam_battle_cage_split",
    "alakazam_neutralization_zone",
    "alakazam_nighttime_mine",
    "crustle_kangaskhan_cage",
    "crustle_kangaskhan_petrel",
    "cynthia_garchomp_roserade",
    "marnie_grimmsnarl_dudunsparce",
    "marnie_grimmsnarl_froslass",
    "marnie_grimmsnarl_tatsugiri",
    "mega_starmie_dusknoir",
}


def load_checkpoint(path: Path) -> dict:
    original = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original  # type: ignore[misc,assignment]


def main() -> int:
    base = load_checkpoint(ROOT / "artifacts/sl0_shared_full/best.pt")
    if base.get("schema_version") != "sl0_shared_checkpoint_v1":
        raise ValueError("invalid SL-0 checkpoint schema")
    base_hash = str(base.get("dataset_sha256", "")).upper()
    if not base_hash:
        raise ValueError("base checkpoint has no dataset hash")

    adapter_root = ROOT / "artifacts/adapters_top10"
    candidates = {path.name for path in adapter_root.iterdir() if path.is_dir()}
    if candidates != EXPECTED:
        raise ValueError(f"adapter candidates mismatch: {sorted(candidates ^ EXPECTED)}")
    for candidate in sorted(EXPECTED):
        checkpoint = load_checkpoint(adapter_root / candidate / "best.pt")
        if checkpoint.get("schema_version") != "deck_adapter_checkpoint_v1":
            raise ValueError(f"invalid adapter schema: {candidate}")
        if checkpoint.get("candidate_id") != candidate:
            raise ValueError(f"adapter candidate mismatch: {candidate}")
        if str(checkpoint.get("base_dataset_sha256", "")).upper() != base_hash:
            raise ValueError(f"adapter base hash mismatch: {candidate}")
        deck_path = ROOT / "data/high_score_decks" / candidate / "deck.csv"
        with deck_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        cards = [
            row
            for row in rows
            if row and any(cell.strip() for cell in row)
        ]
        if len(cards) != 60:
            raise ValueError(f"deck does not contain 60 rows: {candidate}={len(cards)}")

    policy = json.loads(
        (ROOT / "data/high_score_decks/top2_selection_policy.json").read_text(
            encoding="utf-8"
        )
    )
    if policy.get("finalist_count") != 2:
        raise ValueError("Top2 policy does not select two finalists")
    if [row.get("role") for row in policy.get("roles", [])] != [
        "primary",
        "reserve",
    ]:
        raise ValueError("Top2 roles must be primary/reserve")
    if abs(sum(policy.get("score_weights", {}).values()) - 1.0) > 1e-9:
        raise ValueError("Top2 score weights do not sum to one")

    retrain = json.loads(
        (ROOT / "reports/alakazam_battle_cage_split_retrain_eval.json").read_text(
            encoding="utf-8"
        )
    )["candidates"]["alakazam_battle_cage_split"]
    if retrain.get("decision") != "advance" or retrain.get(
        "illegal_predictions"
    ) != 0:
        raise ValueError("verified Alakazam retrain did not pass")

    print(
        json.dumps(
            {
                "ok": True,
                "adapters": len(EXPECTED),
                "decks": len(EXPECTED),
                "roles": ["primary", "reserve"],
                "base_dataset_sha256": base_hash,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
