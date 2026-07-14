"""Run the Top10 stage-B stability screen with one generic action policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_match(
    *, agent0: str, agent1: str, deck0: Path, deck1: Path | None,
    games: int, seed: int, output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable, str(ROOT / "eval/run_match.py"),
        "--agent0", agent0, "--agent1", agent1,
        "--deck0", str(deck0), "--games", str(games),
        "--seed", str(seed), "--out-dir", str(output),
    ]
    if deck1 is not None:
        command.extend(["--deck1", str(deck1)])
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return load(output / "summary.json")


def candidate_result(summary: dict[str, Any], side: int) -> dict[str, int]:
    return {
        "games": int(summary["games"]),
        "wins": int(summary[f"agent{side}_wins"]),
        "losses": int(summary[f"agent{1 - side}_wins"]),
        "draws": int(summary["draws"]),
        "exceptions": int(summary["exceptions"]),
        "illegal_actions": int(summary["illegal_actions"][side]),
        "opponent_illegal_actions": int(summary["illegal_actions"][1 - side]),
    }


def combine(parts: list[dict[str, int]]) -> dict[str, Any]:
    totals = {key: sum(part[key] for part in parts) for key in parts[0]}
    totals["win_rate"] = round(totals["wins"] / totals["games"], 6) if totals["games"] else 0.0
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-games", type=int, default=50)
    parser.add_argument("--random-games", type=int, default=100)
    parser.add_argument("--first-min-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/high_score_deck_selection/stage_b")
    args = parser.parse_args()

    manifest = load(ROOT / "data/high_score_decks/selection_manifest.json")
    candidate_ids = [item["candidate_id"] for item in manifest["ranking_by_exact_replay_wilson_lower_bound"]]
    incumbent = ROOT / "submission/deck.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()

    for candidate_index, candidate_id in enumerate(candidate_ids):
        deck = ROOT / "data/high_score_decks" / candidate_id / "deck.csv"
        candidate_dir = args.out_dir / candidate_id
        base_seed = args.seed + candidate_index * 100
        mirror = run_match(
            agent0="first-min", agent1="first-min", deck0=deck, deck1=deck,
            games=args.mirror_games, seed=base_seed, output=candidate_dir / "mirror",
        )

        matchup_results = {}
        for offset, (name, opponent, games) in enumerate((
            ("random", "random", args.random_games),
            ("first_min", "first-min", args.first_min_games),
        ), 1):
            first_games = games // 2
            second_games = games - first_games
            first = run_match(
                agent0="first-min", agent1=opponent, deck0=deck, deck1=incumbent,
                games=first_games, seed=base_seed + offset, output=candidate_dir / f"vs_{name}_first",
            )
            second = run_match(
                agent0=opponent, agent1="first-min", deck0=incumbent, deck1=deck,
                games=second_games, seed=base_seed + offset, output=candidate_dir / f"vs_{name}_second",
            )
            matchup_results[name] = combine([candidate_result(first, 0), candidate_result(second, 1)])

        stability = combine([candidate_result(mirror, 0)])
        hard_gate_ok = all(
            item["exceptions"] == 0
            and item["illegal_actions"] == 0
            and item["opponent_illegal_actions"] == 0
            for item in [stability, *matchup_results.values()]
        )
        result = {
            "candidate_id": candidate_id,
            "generic_policy": "first-min",
            "mirror": stability,
            "vs_random": matchup_results["random"],
            "vs_first_min": matchup_results["first_min"],
            "hard_gate_ok": hard_gate_ok,
        }
        (candidate_dir / "stage_b_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)

    results.sort(
        key=lambda item: (item["hard_gate_ok"], item["vs_first_min"]["win_rate"], item["vs_random"]["win_rate"]),
        reverse=True,
    )
    report = {
        "schema_version": "high_score_deck_stage_b_v1",
        "policy": "first-min for every candidate; incumbent deck for fixed opponents",
        "side_control": "candidate plays half of non-mirror games from each seat",
        "seed": args.seed,
        "candidate_count": len(results),
        "hard_gate_pass_count": sum(bool(item["hard_gate_ok"]) for item in results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "ranking_reference_only": results,
        "common_policy_priority": [item["candidate_id"] for item in results if item["hard_gate_ok"]][:8],
        "adapter_training_candidates": [item["candidate_id"] for item in results if item["hard_gate_ok"]],
        "decision": "all ten retain Adapter training eligibility; top eight only set experiment priority",
        "ok": len(results) == 10 and all(item["hard_gate_ok"] for item in results),
    }
    (args.out_dir / "stage_b_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "hard_gate_pass_count": report["hard_gate_pass_count"],
        "elapsed_seconds": report["elapsed_seconds"],
        "ok": report["ok"],
    }, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
