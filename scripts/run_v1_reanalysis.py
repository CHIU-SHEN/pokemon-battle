"""Run bounded V1 search on selected target-deck historical decisions."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    parser.add_argument("--queue", type=Path, default=ROOT / "data/reanalysis/v1_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reanalysis/v1_labels.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/reanalysis/v1_labels_summary.json")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=64)
    parser.add_argument("--time-budget", type=float, default=0.08)
    parser.add_argument("--switch-margin", type=float, default=175.0)
    args = parser.parse_args()

    queue = []
    with args.queue.open(encoding="utf-8") as stream:
        for line in stream:
            if len(queue) >= args.max_items:
                break
            queue.append(json.loads(line))
    by_path = defaultdict(dict)
    for item in queue:
        by_path[item["source"]["path"]][int(item["step"])] = item

    labels = []
    failures = []
    started = time.perf_counter()
    for relative_path, wanted in by_path.items():
        path = ROOT / relative_path
        doc = json.loads(path.read_text(encoding="utf-8"))
        ledger = GameLedger()
        manager = SearchManager(
            deck=read_deck(),
            config=SearchConfig(
                enabled=True,
                max_candidates=args.max_candidates,
                particles=args.particles,
                node_budget=args.node_budget,
                time_budget_sec=args.time_budget,
                switch_margin=args.switch_margin,
            ),
        )
        for trace in (doc.get("record") or {}).get("trace") or []:
            observation = trace.get("observation")
            if not isinstance(observation, dict):
                continue
            parsed = parse_observation(observation)
            ledger.update(parsed)
            step = int(trace.get("step", -1))
            item = wanted.get(step)
            if item is None:
                continue
            try:
                manager.stats.last_report = {}
                chosen = manager.choose(observation, parsed, ledger)
                report = dict(manager.stats.last_report or {})
                if not is_legal_action(parsed.select, chosen):
                    raise ValueError(f"V1 returned illegal action {chosen}")
                labels.append({
                    "schema_version": "v1_search_label_v1",
                    "sample_id": item["sample_id"],
                    "game_id": item["game_id"],
                    "split": item["split"],
                    "source": item["source"],
                    "step": step,
                    "priority": item["priority"],
                    "observed_action": item["observed_action"],
                    "v0_action": item["v0_action"],
                    "v1_action": chosen,
                    "v1_changed_v0": chosen != item["v0_action"],
                    "search_used": bool(report),
                    "search": report,
                    "budget": {
                        "max_candidates": args.max_candidates,
                        "particles": args.particles,
                        "node_budget": args.node_budget,
                        "time_budget_sec": args.time_budget,
                        "switch_margin": args.switch_margin,
                    },
                })
            except Exception as exc:
                failures.append({"sample_id": item["sample_id"], "error": f"{type(exc).__name__}: {exc}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in labels:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": "v1_search_label_summary_v1",
        "requested": len(queue),
        "labelled": len(labels),
        "failed": len(failures),
        "search_used": sum(row["search_used"] for row in labels),
        "v1_changed_v0": sum(row["v1_changed_v0"] for row in labels),
        "elapsed_sec": elapsed,
        "labels_per_sec": len(labels) / elapsed if elapsed else None,
        "budget": {
            "max_candidates": args.max_candidates,
            "particles": args.particles,
            "node_budget": args.node_budget,
            "time_budget_sec": args.time_budget,
            "switch_margin": args.switch_margin,
        },
        "failures": failures[:100],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if labels and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
