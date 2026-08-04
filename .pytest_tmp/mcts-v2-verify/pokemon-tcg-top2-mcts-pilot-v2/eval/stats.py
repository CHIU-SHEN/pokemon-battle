#!/usr/bin/env python3
"""Statistical promotion rules for Pokemon TCG AI Battle experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def label_summary(summary: dict[str, Any], baseline_win_rate: float = 0.5) -> dict[str, Any]:
    games = int(summary.get("games", 0) or 0)
    wins = int(summary.get("agent0_wins", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    exceptions = int(summary.get("exceptions", 0) or 0)
    illegal_actions = sum(int(x) for x in summary.get("illegal_actions", [0, 0]))
    p95_decision = float(summary.get("p95_agent_time_sec_per_decision", 0.0) or 0.0)
    low, high = wilson_interval(wins, games)
    win_rate = wins / games if games else 0.0
    delta = win_rate - baseline_win_rate

    if exceptions or illegal_actions:
        label = "淘汰"
        reason = "存在异常或非法动作"
    elif games < 200:
        label = "观察"
        reason = "样本数不足 200，只能 smoke test"
    elif delta > 0.02 and low > baseline_win_rate:
        label = "晋级"
        reason = "胜率提升超过 2%，Wilson 区间下界高于基线"
    elif high < baseline_win_rate:
        label = "淘汰"
        reason = "Wilson 区间上界低于基线"
    else:
        label = "观察"
        reason = "置信区间仍有重叠或提升不足 2%"

    return {
        "agent0": summary.get("agent0"),
        "agent1": summary.get("agent1"),
        "games": games,
        "wins": wins,
        "losses": int(summary.get("agent1_wins", 0) or 0),
        "draws": draws,
        "win_rate": win_rate,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "baseline_win_rate": baseline_win_rate,
        "delta_vs_baseline": delta,
        "exceptions": exceptions,
        "illegal_actions": illegal_actions,
        "p95_agent_time_sec_per_decision": p95_decision,
        "decision": label,
        "decision_reason": reason,
        "source": summary.get("out_dir"),
    }


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="summary.json from eval/run_match.py")
    parser.add_argument("--baseline-win-rate", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = label_summary(load_summary(Path(args.summary)), args.baseline_win_rate)
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

