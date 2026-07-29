#!/usr/bin/env python3
"""Run the resumable Top10 Adapter round-robin and external baseline matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "data" / "high_score_decks"
ADAPTER_ROOT = ROOT / "artifacts" / "adapters_top10"
POLICY_PATH = CANDIDATE_ROOT / "top2_selection_policy.json"
RUN_MATCH = ROOT / "eval" / "run_match.py"

BASELINES = {
    "Random": "random",
    "Exploiter-FirstMin": "first-min",
    "V0-current": "submission",
    "V0-best": str(ROOT / "baselines" / "v0_best" / "main.py"),
    "Sample": str(ROOT / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission" / "main.py"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def candidate_ids() -> list[str]:
    values = sorted(path.name for path in ADAPTER_ROOT.iterdir() if path.is_dir())
    if len(values) != 10:
        raise ValueError(f"expected 10 frozen Adapters, found {len(values)}")
    for candidate in values:
        if not (ADAPTER_ROOT / candidate / "best.pt").is_file():
            raise FileNotFoundError(f"missing Adapter checkpoint: {candidate}")
        if not (CANDIDATE_ROOT / candidate / "deck.csv").is_file():
            raise FileNotFoundError(f"missing deck: {candidate}")
    return values


def available_baselines(requested: list[str]) -> tuple[dict[str, str], list[str]]:
    selected: dict[str, str] = {}
    missing: list[str] = []
    for name in requested:
        if name not in BASELINES:
            raise ValueError(f"unknown baseline {name!r}; available={sorted(BASELINES)}")
        spec = BASELINES[name]
        if spec not in {"random", "first-min", "submission"} and not Path(spec).is_file():
            missing.append(name)
        else:
            selected[name] = spec
    return selected, missing


def internal_tasks(candidates: list[str], games: int, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(itertools.combinations(candidates, 2)):
        pair_dir = out_dir / "internal" / f"{left}__vs__{right}"
        common_seed = seed + pair_index
        tasks.extend(
            [
                {
                    "id": f"internal:{left}:{right}:left_first",
                    "kind": "internal",
                    "candidate0": left,
                    "candidate1": right,
                    "agent0": f"adapter:{left}",
                    "agent1": f"adapter:{right}",
                    "games": games,
                    "seed": common_seed,
                    "out_dir": pair_dir / "left_first",
                },
                {
                    "id": f"internal:{left}:{right}:right_first",
                    "kind": "internal",
                    "candidate0": right,
                    "candidate1": left,
                    "agent0": f"adapter:{right}",
                    "agent1": f"adapter:{left}",
                    "games": games,
                    "seed": common_seed,
                    "out_dir": pair_dir / "right_first",
                },
            ]
        )
    return tasks


def external_tasks(
    candidates: list[str], baselines: dict[str, str], games: int, seed: int, out_dir: Path
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        for baseline_index, (baseline_name, baseline_spec) in enumerate(baselines.items()):
            match_seed = seed + 10_000 + candidate_index * 100 + baseline_index
            match_dir = out_dir / "external" / candidate / f"vs_{baseline_name}"
            tasks.extend(
                [
                    {
                        "id": f"external:{candidate}:{baseline_name}:candidate_first",
                        "kind": "external",
                        "candidate": candidate,
                        "baseline": baseline_name,
                        "candidate_side": 0,
                        "agent0": f"adapter:{candidate}",
                        "agent1": baseline_spec,
                        "games": games,
                        "seed": match_seed,
                        "out_dir": match_dir / "candidate_first",
                    },
                    {
                        "id": f"external:{candidate}:{baseline_name}:candidate_second",
                        "kind": "external",
                        "candidate": candidate,
                        "baseline": baseline_name,
                        "candidate_side": 1,
                        "agent0": baseline_spec,
                        "agent1": f"adapter:{candidate}",
                        "games": games,
                        "seed": match_seed,
                        "out_dir": match_dir / "candidate_second",
                    },
                ]
            )
    return tasks


def valid_cached_summary(task: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(task["out_dir"]) / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        summary.get("agent0") != task["agent0"]
        or summary.get("agent1") != task["agent1"]
        or int(summary.get("games", -1)) != task["games"]
        or int(summary.get("seed", -1)) != task["seed"]
    ):
        return None
    return summary


def run_task(task: dict[str, Any], resume: bool) -> tuple[dict[str, Any], bool]:
    cached = valid_cached_summary(task) if resume else None
    if cached is not None:
        return cached, True

    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_MATCH),
        "--agent0",
        task["agent0"],
        "--agent1",
        task["agent1"],
        "--games",
        str(task["games"]),
        "--seed",
        str(task["seed"]),
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PTCG_ADAPTER_DEVICE", "cpu")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"match failed ({task['id']}), see {log_path}")
    summary = read_json(out_dir / "summary.json")
    if valid_cached_summary(task) is None:
        raise RuntimeError(f"match wrote an invalid summary ({task['id']})")
    return summary, False


def candidate_score(summary: dict[str, Any], side: int) -> tuple[float, int, float]:
    games = int(summary["games"])
    wins = int(summary[f"agent{side}_wins"])
    draws = int(summary["draws"])
    return wins + 0.5 * draws, games, float(summary.get("p95_agent_time_sec_per_decision", 0.0))


def wilson_interval(points: float, games: int, z: float = 1.96) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    p = points / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def build_report(
    candidates: list[str], tasks: list[dict[str, Any]], results: dict[str, Any], policy: dict[str, Any],
    missing_baselines: list[str], latency_gate: float, started_at: str, elapsed_seconds: float,
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {
        candidate: {
            "internal": {}, "external": {}, "exceptions": 0, "illegal_actions": 0,
            "max_p95_decision_seconds": 0.0,
        }
        for candidate in candidates
    }

    def add(candidate: str, group: str, opponent: str, summary: dict[str, Any], side: int) -> None:
        points, games, latency = candidate_score(summary, side)
        bucket = aggregates[candidate][group].setdefault(opponent, {"points": 0.0, "games": 0})
        bucket["points"] += points
        bucket["games"] += games
        aggregates[candidate]["exceptions"] += int(summary.get("exceptions", 0))
        illegal = summary.get("illegal_actions", [0, 0])
        aggregates[candidate]["illegal_actions"] += int(illegal[side])
        aggregates[candidate]["max_p95_decision_seconds"] = max(
            aggregates[candidate]["max_p95_decision_seconds"], latency
        )

    for task in tasks:
        summary = results.get(task["id"])
        if summary is None:
            continue
        if task["kind"] == "internal":
            add(task["candidate0"], "internal", task["candidate1"], summary, 0)
            add(task["candidate1"], "internal", task["candidate0"], summary, 1)
        else:
            add(task["candidate"], "external", task["baseline"], summary, task["candidate_side"])

    weights = policy["score_weights"]
    ranking = []
    for candidate, value in aggregates.items():
        for group in ("internal", "external"):
            for opponent_value in value[group].values():
                games = opponent_value["games"]
                opponent_value["score"] = opponent_value["points"] / games if games else 0.0

        internal_points = sum(item["points"] for item in value["internal"].values())
        internal_games = sum(item["games"] for item in value["internal"].values())
        external_points = sum(item["points"] for item in value["external"].values())
        external_games = sum(item["games"] for item in value["external"].values())
        internal_score = internal_points / internal_games if internal_games else 0.0
        external_score = external_points / external_games if external_games else 0.0
        opponent_scores = [
            item["score"] for group in ("internal", "external") for item in value[group].values()
        ]
        worst_score = min(opponent_scores) if opponent_scores else 0.0
        gate_pass = value["exceptions"] == 0 and value["illegal_actions"] == 0
        latency_score = max(0.0, 1.0 - value["max_p95_decision_seconds"] / latency_gate)
        stability_score = latency_score if gate_pass else 0.0
        composite = (
            weights["internal_round_robin"] * internal_score
            + weights["external_baseline_score"] * external_score
            + weights["worst_matchup_score"] * worst_score
            + weights["stability_and_latency"] * stability_score
        )
        value.update(
            {
                "internal_score": internal_score,
                "internal_games": internal_games,
                "internal_wilson_95": wilson_interval(internal_points, internal_games),
                "external_score": external_score,
                "external_games": external_games,
                "worst_matchup_score": worst_score,
                "stability_and_latency_score": stability_score,
                "hard_gate_pass": gate_pass,
                "composite_score": composite,
            }
        )
        ranking.append({"candidate": candidate, **value})

    ranking.sort(key=lambda row: (row["hard_gate_pass"], row["composite_score"]), reverse=True)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index

    completed = len(results)
    return {
        "schema_version": "top2_adapter_arena_preliminary_v1",
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "engine_seed_controlled": False,
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "score_weights": weights,
        "missing_baselines": missing_baselines,
        "tasks_total": len(tasks),
        "tasks_completed": completed,
        "complete": completed == len(tasks),
        "games_completed": sum(int(item.get("games", 0)) for item in results.values()),
        "ranking": ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stages", default="internal,external", help="internal, external, or both")
    parser.add_argument(
        "--candidate-subset",
        default="",
        help="comma-separated candidate IDs for playoffs/finals; default uses all 10",
    )
    parser.add_argument(
        "--baselines", default="Random,Exploiter-FirstMin,V0-current,V0-best,Sample"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "top2_adapter_arena")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "top2_arena_report.json")
    parser.add_argument("--latency-sec", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0, help="development smoke cap; 0 runs all tasks")
    args = parser.parse_args()
    if args.games_per_seat <= 0 or args.workers <= 0:
        raise ValueError("games and workers must be positive")

    stages = {value.strip() for value in args.stages.split(",") if value.strip()}
    if not stages or not stages <= {"internal", "external"}:
        raise ValueError("--stages must contain internal and/or external")
    requested_baselines = [value.strip() for value in args.baselines.split(",") if value.strip()]
    baselines, missing_baselines = available_baselines(requested_baselines)
    all_candidates = candidate_ids()
    if args.candidate_subset:
        requested_candidates = [value.strip() for value in args.candidate_subset.split(",") if value.strip()]
        unknown_candidates = sorted(set(requested_candidates) - set(all_candidates))
        if unknown_candidates:
            raise ValueError(f"unknown candidates: {unknown_candidates}")
        candidates = list(dict.fromkeys(requested_candidates))
        if len(candidates) < 2:
            raise ValueError("--candidate-subset must contain at least two distinct candidates")
    else:
        candidates = all_candidates
    policy = read_json(POLICY_PATH)
    if not math.isclose(sum(float(value) for value in policy["score_weights"].values()), 1.0):
        raise ValueError("selection policy weights must sum to 1")

    tasks: list[dict[str, Any]] = []
    if "internal" in stages:
        tasks.extend(internal_tasks(candidates, args.games_per_seat, args.seed, args.out_dir))
    if "external" in stages:
        tasks.extend(external_tasks(candidates, baselines, args.games_per_seat, args.seed, args.out_dir))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoint.json"
    started_at = time.strftime("%Y%m%d_%H%M%S")
    started = time.time()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    print(
        json.dumps(
            {
                "candidates": len(candidates), "tasks": len(tasks), "workers": args.workers,
                "games_per_seat": args.games_per_seat, "missing_baselines": missing_baselines,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(run_task, task, not args.no_resume): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                summary, cached = future.result()
                results[task["id"]] = summary
                status = "cached" if cached else "done"
            except Exception as exc:  # preserve other completed matches and make the run resumable
                failures[task["id"]] = str(exc)
                status = f"failed: {exc}"
            write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": "top2_adapter_arena_checkpoint_v1",
                    "tasks_total": len(tasks),
                    "tasks_completed": len(results),
                    "failures": failures,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            print(f"[{len(results) + len(failures)}/{len(tasks)}] {task['id']}: {status}", flush=True)

    report = build_report(
        candidates, tasks, results, policy, missing_baselines, args.latency_sec,
        started_at, time.time() - started,
    )
    report["failures"] = failures
    write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "complete": report["complete"], "tasks_completed": report["tasks_completed"],
                "games_completed": report["games_completed"], "failures": len(failures),
                "report": str(args.report),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["complete"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
