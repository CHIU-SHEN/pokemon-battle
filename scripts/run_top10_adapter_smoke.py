#!/usr/bin/env python3
"""Run the required Random and mirror smoke matrix for all frozen Adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "data" / "high_score_decks"
ADAPTER_ROOT = ROOT / "artifacts" / "adapters_top10"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deck_hash(candidate: str) -> str:
    cards = [
        int(line)
        for line in (CANDIDATE_ROOT / candidate / "deck.csv").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{candidate} deck has {len(cards)} cards")
    canonical = "\n".join(str(card_id) for card_id in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_match(candidate: str, games: int, seed: int, out_dir: Path, mirror: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "eval" / "run_match.py"),
        "--agent0",
        f"adapter:{candidate}",
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if mirror:
        cmd.append("--mirror")
    else:
        cmd.extend(["--agent1", "random"])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return read_json(out_dir / "summary.json")


def gate(summary: dict[str, Any], expected_hash: str, games: int, mirror: bool, latency_sec: float) -> list[str]:
    problems = []
    if int(summary.get("games", -1)) != games:
        problems.append("wrong_game_count")
    if int(summary.get("exceptions", -1)) != 0:
        problems.append("exceptions")
    if sum(int(value) for value in summary.get("illegal_actions", [])) != 0:
        problems.append("illegal_actions")
    if float(summary.get("p95_agent_time_sec_per_decision", latency_sec + 1)) > latency_sec:
        problems.append("latency")
    if summary.get("deck0_sha256") != expected_hash or not summary.get("deck0_consistent"):
        problems.append("deck0_binding")
    if mirror and (summary.get("deck1_sha256") != expected_hash or not summary.get("deck1_consistent")):
        problems.append("deck1_binding")
    model_calls = int((summary.get("action_sources") or [{}, {}])[0].get("adapter", 0))
    if model_calls <= 0:
        problems.append("no_adapter_actions")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10, help="games per Random/mirror matchup")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--latency-sec", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "top2_adapter_smoke")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "top10_adapter_online_smoke.json")
    args = parser.parse_args()
    if args.games <= 0:
        raise ValueError("--games must be positive")

    candidates = sorted(path.name for path in ADAPTER_ROOT.iterdir() if path.is_dir())
    if len(candidates) != 10:
        raise ValueError(f"expected 10 frozen Adapters, found {len(candidates)}")
    started = time.time()
    results: dict[str, Any] = {}
    for offset, candidate in enumerate(candidates):
        expected_hash = deck_hash(candidate)
        random_summary = run_match(
            candidate, args.games, args.seed + offset, args.out_dir / candidate / "vs_random", mirror=False
        )
        mirror_summary = run_match(
            candidate, args.games, args.seed + offset, args.out_dir / candidate / "mirror", mirror=True
        )
        problems = [
            *[f"random:{value}" for value in gate(random_summary, expected_hash, args.games, False, args.latency_sec)],
            *[f"mirror:{value}" for value in gate(mirror_summary, expected_hash, args.games, True, args.latency_sec)],
        ]
        results[candidate] = {
            "decision": "pass" if not problems else "fail",
            "problems": problems,
            "deck_sha256": expected_hash,
            "vs_random": random_summary,
            "mirror": mirror_summary,
        }
        print(f"{candidate}: {results[candidate]['decision']} {problems}", flush=True)

    failed = [candidate for candidate, value in results.items() if value["decision"] != "pass"]
    report = {
        "schema_version": "top10_adapter_online_smoke_v1",
        "games_per_matchup": args.games,
        "matchups": 20,
        "total_games": args.games * 20,
        "seed": args.seed,
        "engine_seed_controlled": False,
        "latency_gate_seconds": args.latency_sec,
        "elapsed_seconds": time.time() - started,
        "candidates": results,
        "passed": len(results) - len(failed),
        "failed": failed,
        "decision": "pass" if not failed else "fail",
    }
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("decision", "passed", "failed", "total_games", "elapsed_seconds")}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

