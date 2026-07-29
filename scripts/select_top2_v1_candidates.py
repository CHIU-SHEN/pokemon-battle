#!/usr/bin/env python3
"""Select one branch's train-only low-confidence/loss rollout states for V1 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rank_candidates(rows: list[dict], *, deck_id: str, max_items: int) -> list[dict]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("deck_id") == deck_id and row.get("split") == "train"
    ]
    for row in eligible:
        row["priority"] = (
            100.0 * (row.get("game_result") == "loss")
            + 10.0 * (1.0 - float(row.get("confidence", 1.0)))
            + float(row.get("entropy", 0.0))
        )
    eligible.sort(key=lambda row: (-row["priority"], str(row.get("sample_id", ""))))
    return eligible[:max_items]


def load_rows(path: Path) -> list[dict]:
    rows = []
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    for source in files:
        doc = json.loads(source.read_text(encoding="utf-8"))
        if doc.get("schema_version") != "top2_rl_rollout_v1":
            continue
        learner_side = int(doc["learner_side"])
        result = int(doc["record"]["result"])
        game_result = "draw" if result == 2 else "win" if result == learner_side else "loss"
        for index, decision in enumerate(doc.get("decisions") or []):
            row = dict(decision)
            row.update({
                "sample_id": f"{doc['game_id']}:{index:04d}",
                "deck_id": doc["deck_id"],
                "candidate_id": doc["candidate_id"],
                "role": doc["role"],
                "game_result": game_result,
                "source_path": str(source),
            })
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()
    selected = rank_candidates(load_rows(args.rollouts.resolve()), deck_id=args.deck_id, max_items=args.max_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"schema_version": "top2_v1_candidate_queue_v1", "deck_id": args.deck_id, "selected": len(selected), "train_only": True}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
