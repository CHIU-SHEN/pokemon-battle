#!/usr/bin/env python3
"""Run bounded V1 search on a branch-bound Top2 candidate queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
for path in (ROOT, SUBMISSION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent.belief import read_deck  # noqa: E402
from agent.fallback import is_legal_action  # noqa: E402
from agent.parser import GameLedger, parse_observation  # noqa: E402
from agent.search import SearchConfig, SearchManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=64)
    parser.add_argument("--time-budget", type=float, default=0.08)
    args = parser.parse_args()
    queue = []
    with args.queue.open(encoding="utf-8") as stream:
        for line in stream:
            if len(queue) >= args.max_items:
                break
            item = json.loads(line)
            if item.get("deck_id") != args.deck_id or item.get("split") != "train":
                raise ValueError("V1 queue crosses deck_id or includes holdout data")
            queue.append(item)
    manager = SearchManager(
        deck=read_deck(str(args.deck)),
        config=SearchConfig(enabled=True, max_candidates=args.max_candidates, particles=args.particles, node_budget=args.node_budget, time_budget_sec=args.time_budget),
    )
    labels = []
    failures = []
    started = time.perf_counter()
    for item in queue:
        try:
            observation = item["observation"]
            parsed = parse_observation(observation)
            ledger = GameLedger()
            ledger.update(parsed)
            manager.stats.last_report = {}
            chosen = manager.choose(observation, parsed, ledger)
            if not is_legal_action(parsed.select, chosen):
                raise ValueError(f"V1 returned illegal action {chosen}")
            labels.append({
                "schema_version": "top2_v1_search_label_v1", "sample_id": item["sample_id"], "game_id": item["game_id"],
                "deck_id": args.deck_id, "split": "train", "observed_action": [item["action"]], "v1_action": chosen,
                "v1_changed_behavior": chosen != [item["action"]], "search": dict(manager.stats.last_report or {}),
            })
        except Exception as exc:
            failures.append({"sample_id": item.get("sample_id"), "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in labels:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"schema_version": "top2_v1_search_summary_v1", "deck_id": args.deck_id, "requested": len(queue), "labelled": len(labels), "failed": len(failures), "elapsed_sec": time.perf_counter() - started, "failures": failures[:100]}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if labels and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
