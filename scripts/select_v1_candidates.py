"""Rank target-deck bad-case decisions for bounded V1 search reanalysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCHABLE_CONTEXTS = {0, 3, 4, 7, 21, 22, 35, 37, 41}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/bad_case_decisions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reanalysis/v1_candidates.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/reanalysis/v1_candidates_summary.json")
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()

    ranked = []
    eligible = 0
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            v0 = row["teacher"].get("v0_action")
            select = row["select"]
            if v0 is None or row["quality"].get("forced_single_option"):
                continue
            if select["option_count"] <= 1 or select["context"] not in SEARCHABLE_CONTEXTS:
                continue
            if select["min_count"] != select["max_count"] and select["max_count"] > 2:
                continue
            eligible += 1
            disagree = v0 != row["observed_action"]
            many_options = select["option_count"] >= 5
            multi_select = select["max_count"] > 1
            late_turn = row["turn"] >= 8
            loss = row["game_result"] == "loss"
            priority = (
                100.0 * disagree
                + 8.0 * many_options
                + 5.0 * multi_select
                + 3.0 * late_turn
                + 2.0 * loss
                + min(select["option_count"], 20) / 20.0
                + min(row["turn"], 50) / 100.0
            )
            ranked.append({
                "sample_id": row["sample_id"],
                "game_id": row["game_id"],
                "source": row["source"],
                "step": row["step"],
                "split": row["split"],
                "priority": priority,
                "reason": {
                    "v0_observed_disagree": disagree,
                    "many_options": many_options,
                    "multi_select": multi_select,
                    "late_turn": late_turn,
                    "loss": loss,
                },
                "select": select,
                "observed_action": row["observed_action"],
                "v0_action": v0,
            })
    ranked.sort(key=lambda x: (-x["priority"], x["sample_id"]))
    selected = ranked[: args.max_items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "v1_candidate_queue_v1",
        "eligible": eligible,
        "selected": len(selected),
        "true_v0_observed_disagreements": sum(x["reason"]["v0_observed_disagree"] for x in selected),
        "selection_rule": "target submission only; searchable contexts; exclude forced choices; prioritize real disagreement, option complexity, late turns, and losses",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())

