#!/usr/bin/env python3
"""Run local Pokemon TCG AI Battle evaluations and write match metrics."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from typing import Callable
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


Agent = Callable[[dict | None], list[int]]


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def read_deck(path: Path = SUBMISSION_DIR / "deck.csv") -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def deck_sha256(deck: list[int]) -> str:
    canonical = "\n".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def with_deck(agent: Agent, deck: list[int] | None) -> Agent:
    if deck is None:
        return agent

    def wrapped(obs_dict: dict | None) -> list[int]:
        if obs_dict is None or obs_dict.get("select") is None:
            return list(deck)
        return agent(obs_dict)

    return wrapped


def random_agent(obs_dict: dict | None) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def first_min_agent(obs_dict: dict | None) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    return list(range(obs.select.minCount))


def load_agent(spec: str) -> Agent:
    if spec == "random":
        return random_agent
    if spec == "first-min":
        return first_min_agent
    if spec == "submission":
        spec = str(SUBMISSION_DIR / "main.py")

    path = Path(spec).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"agent file not found: {path}")
    module_name = f"agent_{abs(hash(path))}"
    import_spec = importlib.util.spec_from_file_location(module_name, path)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"cannot import agent from {path}")
    module = importlib.util.module_from_spec(import_spec)
    sys.path.insert(0, str(path.parent))
    try:
        sys.modules[module_name] = module
        with pushd(path.parent):
            import_spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} does not expose agent(obs_dict)")
    agent_dir = path.parent
    raw_agent = module.agent

    def wrapped(obs_dict: dict | None) -> list[int]:
        with pushd(agent_dir):
            return raw_agent(obs_dict)

    if hasattr(module, "action_source"):
        wrapped.action_source = module.action_source

    return wrapped


def validate_action(obs_dict: dict, action: list[int]) -> str | None:
    obs = to_observation_class(obs_dict)
    select = obs.select
    if select is None:
        if not isinstance(action, list) or len(action) != 60:
            return "deck selection must return a 60-card list"
        if not all(isinstance(x, int) for x in action):
            return "deck selection must contain only integers"
        return None

    if not isinstance(action, list):
        return "action is not a list"
    if not all(isinstance(x, int) for x in action):
        return "action contains non-integer values"
    if not (select.minCount <= len(action) <= select.maxCount):
        return f"action length {len(action)} outside [{select.minCount}, {select.maxCount}]"
    if len(set(action)) != len(action):
        return "action contains duplicate indices"
    option_count = len(select.option)
    if any(x < 0 or x >= option_count for x in action):
        return f"action index outside [0, {option_count})"
    return None


def call_agent(agent: Agent, obs_dict: dict | None) -> tuple[list[int], float]:
    start = time.perf_counter()
    action = agent(obs_dict)
    return action, time.perf_counter() - start


def play_game(
    agent0: Agent,
    agent1: Agent,
    max_steps: int,
    fixtures: list[dict] | None = None,
    fixture_limit: int = 0,
    trace: bool = False,
) -> dict:
    record = {
        "result": None,
        "steps": 0,
        "selection_count": 0,
        "agent_time": [0.0, 0.0],
        "agent_decision_times": [[], []],
        "agent_calls": [0, 0],
        "illegal_actions": [0, 0],
        "action_sources": [{}, {}],
        "exceptions": [],
        "draw_by_max_steps": False,
        "trace": [] if trace else None,
    }
    battle_open = False
    try:
        deck0, deck0_time = call_agent(agent0, {"select": None, "logs": [], "current": None})
        deck1, deck1_time = call_agent(agent1, {"select": None, "logs": [], "current": None})
        record["agent_time"][0] += deck0_time
        record["agent_time"][1] += deck1_time
        record["agent_decision_times"][0].append(deck0_time)
        record["agent_decision_times"][1].append(deck1_time)
        record["agent_calls"][0] += 1
        record["agent_calls"][1] += 1

        obs, start_data = battle_start(deck0, deck1)
        battle_open = True
        if start_data.errorType != 0:
            raise RuntimeError(
                f"battle_start failed: errorType={start_data.errorType}, "
                f"errorPlayer={start_data.errorPlayer}"
            )

        while True:
            obs_obj = to_observation_class(obs)
            if obs_obj.current is not None and obs_obj.current.result != -1:
                record["result"] = obs_obj.current.result
                break

            player_idx = obs_obj.current.yourIndex
            if fixtures is not None and len(fixtures) < fixture_limit:
                fixtures.append(obs)
            action, elapsed = call_agent(agent0 if player_idx == 0 else agent1, obs)
            active_agent = agent0 if player_idx == 0 else agent1
            source_reader = getattr(active_agent, "action_source", None)
            source = str(source_reader()) if source_reader is not None else "unreported"
            record["action_sources"][player_idx][source] = record["action_sources"][player_idx].get(source, 0) + 1
            record["agent_time"][player_idx] += elapsed
            record["agent_decision_times"][player_idx].append(elapsed)
            record["agent_calls"][player_idx] += 1
            record["selection_count"] += 1

            if trace:
                obs_obj = to_observation_class(obs)
                select = obs_obj.select
                record["trace"].append(
                    {
                        "step": record["steps"],
                        "player": player_idx,
                        "elapsed_sec": elapsed,
                        "action": action,
                        "select_type": getattr(select.type, "value", select.type) if select else None,
                        "select_context": getattr(select.context, "value", select.context) if select else None,
                        "min_count": select.minCount if select else None,
                        "max_count": select.maxCount if select else None,
                        "option_count": len(select.option) if select else None,
                        "observation": obs,
                    }
                )

            validation_error = validate_action(obs, action)
            if validation_error is not None:
                record["illegal_actions"][player_idx] += 1
                if trace and record["trace"]:
                    record["trace"][-1]["validation_error"] = validation_error
                raise ValueError(f"player {player_idx} illegal action: {validation_error}")

            obs = battle_select(action)
            record["steps"] += 1
            if record["steps"] > max_steps:
                record["result"] = 2
                record["draw_by_max_steps"] = True
                break
    except Exception as exc:  # keep evaluation going and report failures
        record["result"] = 2
        record["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
    finally:
        if battle_open:
            battle_finish()
    return record


def compact_game_record(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "trace"}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def summarize(records: list[dict], agent0_name: str, agent1_name: str, seed: int) -> dict:
    result_counts = {
        "agent0_wins": sum(1 for r in records if r["result"] == 0),
        "agent1_wins": sum(1 for r in records if r["result"] == 1),
        "draws": sum(1 for r in records if r["result"] == 2),
    }
    game_times = [sum(r["agent_time"]) for r in records]
    decision_times = [t for r in records for side in r.get("agent_decision_times", []) for t in side]
    steps = [r["steps"] for r in records]
    selections = [r["selection_count"] for r in records]
    exceptions = sum(len(r["exceptions"]) for r in records)
    illegal0 = sum(r["illegal_actions"][0] for r in records)
    illegal1 = sum(r["illegal_actions"][1] for r in records)
    action_sources = [{}, {}]
    for record in records:
        for side in (0, 1):
            for source, count in record.get("action_sources", [{}, {}])[side].items():
                action_sources[side][source] = action_sources[side].get(source, 0) + int(count)
    total = len(records)
    non_draw = max(1, total - result_counts["draws"])
    return {
        "agent0": agent0_name,
        "agent1": agent1_name,
        "seed": seed,
        "engine_seed_controlled": False,
        "games": total,
        **result_counts,
        "agent0_win_rate_all": result_counts["agent0_wins"] / total if total else 0.0,
        "agent1_win_rate_all": result_counts["agent1_wins"] / total if total else 0.0,
        "agent0_win_rate_non_draw": result_counts["agent0_wins"] / non_draw,
        "agent1_win_rate_non_draw": result_counts["agent1_wins"] / non_draw,
        "avg_steps": statistics.fmean(steps) if steps else 0.0,
        "avg_selection_count": statistics.fmean(selections) if selections else 0.0,
        "avg_agent_time_sec_per_game": statistics.fmean(game_times) if game_times else 0.0,
        "p95_agent_time_sec_per_game": percentile(game_times, 0.95),
        "avg_agent_time_sec_per_decision": statistics.fmean(decision_times) if decision_times else 0.0,
        "p95_agent_time_sec_per_decision": percentile(decision_times, 0.95),
        "draw_by_max_steps": sum(1 for r in records if r["draw_by_max_steps"]),
        "exceptions": exceptions,
        "illegal_actions": [illegal0, illegal1],
        "action_sources": action_sources,
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent0", default="submission", help="submission, random, first-min, or /path/to/main.py")
    parser.add_argument("--agent1", default="random", help="submission, random, first-min, or /path/to/main.py")
    parser.add_argument("--deck0", default=None, help="optional deck.csv override for agent0")
    parser.add_argument("--deck1", default=None, help="optional deck.csv override for agent1")
    parser.add_argument("--mirror", action="store_true", help="run agent0 against itself")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=20260706,
        help="seed for Python-side agents only; the bundled battle engine exposes no RNG seed control",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--out-dir", default=None, help="default: experiments/YYYYMMDD_HHMMSS_<matchup>")
    parser.add_argument("--save-fixtures", type=int, default=0, help="save up to N observations for regression tests")
    parser.add_argument("--bad-case-dir", default=None, help="write replayable bad cases to this directory")
    parser.add_argument("--high-latency-sec", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.mirror:
        args.agent1 = args.agent0
        if args.deck1 is None:
            args.deck1 = args.deck0

    deck0 = read_deck(Path(args.deck0)) if args.deck0 else None
    deck1 = read_deck(Path(args.deck1)) if args.deck1 else None
    agent0 = with_deck(load_agent(args.agent0), deck0)
    agent1 = with_deck(load_agent(args.agent1), deck1)
    started = time.strftime("%Y%m%d_%H%M%S")
    matchup = f"{Path(args.agent0).stem}_vs_{Path(args.agent1).stem}".replace(os.sep, "_")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "experiments" / f"{started}_{matchup}"
    fixtures: list[dict] = []
    collect_trace = args.bad_case_dir is not None
    records = [
        play_game(agent0, agent1, args.max_steps, fixtures, args.save_fixtures, trace=collect_trace)
        for _ in range(args.games)
    ]
    summary = summarize(records, args.agent0, args.agent1, args.seed)
    summary["deck0"] = args.deck0
    summary["deck1"] = args.deck1
    summary["deck0_sha256"] = deck_sha256(deck0) if deck0 else None
    summary["deck1_sha256"] = deck_sha256(deck1) if deck1 else None
    summary["started_at"] = started
    summary["out_dir"] = str(out_dir)

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "games.json", [compact_game_record(r) for r in records])
    if args.bad_case_dir:
        bad_case_dir = Path(args.bad_case_dir)
        if not bad_case_dir.is_absolute():
            bad_case_dir = PROJECT_ROOT / bad_case_dir
        saved_cases = []
        for game_index, record in enumerate(records):
            decision_times = [
                item.get("elapsed_sec", 0.0)
                for item in (record.get("trace") or [])
                if item.get("player") == 0
            ]
            reasons = []
            if record["result"] == 1:
                reasons.append("agent0_loss")
            if record["exceptions"]:
                reasons.append("exception")
            if record["draw_by_max_steps"]:
                reasons.append("max_steps_draw")
            if sum(record["illegal_actions"]):
                reasons.append("illegal_action")
            if decision_times and max(decision_times) >= args.high_latency_sec:
                reasons.append("high_latency")
            if reasons:
                case_id = f"{started}_{game_index:05d}_{'_'.join(reasons)}"
                case = {
                    "case_id": case_id,
                    "matchup": {"agent0": args.agent0, "agent1": args.agent1},
                    "seed": args.seed,
                    "game_index": game_index,
                    "reasons": reasons,
                    "record": record,
                }
                case_path = bad_case_dir / f"{case_id}.json"
                write_json(case_path, case)
                saved_cases.append(str(case_path))
        summary["bad_case_dir"] = str(bad_case_dir)
        summary["bad_cases_saved"] = len(saved_cases)
        summary["bad_case_paths"] = saved_cases[:20]
        write_json(out_dir / "summary.json", summary)
    if args.save_fixtures:
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "observations.json"
        write_json(fixture_path, fixtures[: args.save_fixtures])
        summary["fixtures_path"] = str(fixture_path)
        write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["exceptions"] == 0 and sum(summary["illegal_actions"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
