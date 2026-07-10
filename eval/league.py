#!/usr/bin/env python3
"""Run fixed-baseline league matrices for M2 experiment discipline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from stats import label_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_AGENT = PROJECT_ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "main.py"
V0_BEST_AGENT = PROJECT_ROOT / "baselines" / "v0_best" / "main.py"

BASELINES = {
    "Random": "random",
    "Sample": str(SAMPLE_AGENT),
    "Exploiter-FirstMin": "first-min",
    "V0-current": "submission",
    "V0-best": str(V0_BEST_AGENT),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_match(agent0: str, agent1: str, games: int, seed: int, out_dir: Path, bad_case_dir: Path | None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "run_match.py"),
        "--agent0",
        agent0,
        "--agent1",
        agent1,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if bad_case_dir is not None:
        cmd.extend(["--bad-case-dir", str(bad_case_dir)])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return load_json(out_dir / "summary.json")


def markdown_matrix(matrix: dict[str, dict[str, Any]]) -> str:
    headers = ["Candidate \\ Baseline", *matrix.keys()]
    row = ["candidate"]
    for name, result in matrix.items():
        row.append(
            f"{result['agent0_win_rate_all']:.3f} "
            f"({result['agent0_wins']}-{result['agent1_wins']}-{result['draws']})"
        )
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---", *["---:" for _ in matrix]]) + " |",
            "| " + " | ".join(row) + " |",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="submission")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--baselines", default="Random,Sample,Exploiter-FirstMin")
    parser.add_argument("--include-self", action="store_true", help="also evaluate against V0-current")
    parser.add_argument("--save-bad-cases", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    started = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "experiments" / f"{started}_league"
    selected = [name.strip() for name in args.baselines.split(",") if name.strip()]
    if args.include_self and "V0-current" not in selected:
        selected.append("V0-current")
    unknown = [name for name in selected if name not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}; available={sorted(BASELINES)}")

    matrix: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    bad_case_root = PROJECT_ROOT / "logs" / "bad_cases" / started if args.save_bad_cases else None
    for offset, name in enumerate(selected):
        match_dir = out_dir / f"candidate_vs_{name}"
        bad_case_dir = bad_case_root / name if bad_case_root else None
        summary = run_match(args.candidate, BASELINES[name], args.games, args.seed + offset, match_dir, bad_case_dir)
        matrix[name] = summary
        labels[name] = label_summary(summary, baseline_win_rate=0.5)
        if BASELINES[name] == args.candidate:
            labels[name]["decision"] = "观察"
            labels[name]["decision_reason"] = "同源 mirror/sanity 对局，不能作为晋级证据"

    report = {
        "candidate": args.candidate,
        "games_per_matchup": args.games,
        "seed": args.seed,
        "baselines": selected,
        "matrix": matrix,
        "labels": labels,
        "markdown_matrix": markdown_matrix(matrix),
        "bad_case_root": str(bad_case_root) if bad_case_root else None,
    }
    write_json(out_dir / "league_report.json", report)
    print(report["markdown_matrix"])
    print(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"league_report: {out_dir / 'league_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
