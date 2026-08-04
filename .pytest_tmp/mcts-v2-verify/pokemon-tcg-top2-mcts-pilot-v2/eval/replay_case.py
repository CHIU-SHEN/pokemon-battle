#!/usr/bin/env python3
"""Replay a saved bad-case trace and compare agent choices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from cg.api import to_observation_class  # noqa: E402


Agent = Callable[[dict | None], list[int]]


def push_agent_path(path: Path):
    sys.path.insert(0, str(path.parent))


def load_agent(spec: str) -> Agent:
    if spec == "submission":
        spec = str(SUBMISSION_DIR / "main.py")
    elif spec == "random":
        def random_like(obs_dict):
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                with (SUBMISSION_DIR / "deck.csv").open() as f:
                    return [int(x.strip()) for x in f if x.strip()]
            return list(range(obs.select.minCount))
        return random_like

    path = Path(spec).expanduser().resolve()
    module_name = f"replay_agent_{abs(hash(path))}"
    import_spec = importlib.util.spec_from_file_location(module_name, path)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(import_spec)
    old_cwd = Path.cwd()
    push_agent_path(path)
    os.chdir(path.parent)
    try:
        import_spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)
    raw_agent = module.agent

    def wrapped(obs_dict):
        old = Path.cwd()
        os.chdir(path.parent)
        try:
            return raw_agent(obs_dict)
        finally:
            os.chdir(old)

    return wrapped


def validate(obs_dict: dict, action: list[int]) -> str:
    obs = to_observation_class(obs_dict)
    select = obs.select
    if select is None:
        return "deck" if len(action) == 60 else "invalid deck"
    ok = (
        isinstance(action, list)
        and all(isinstance(x, int) for x in action)
        and select.minCount <= len(action) <= select.maxCount
        and len(set(action)) == len(action)
        and all(0 <= x < len(select.option) for x in action)
    )
    return "legal" if ok else "illegal"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_json")
    parser.add_argument("--agent", action="append", default=["submission"], help="agent spec; can be repeated")
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with Path(args.case_json).open("r", encoding="utf-8") as f:
        case = json.load(f)
    agents = [(spec, load_agent(spec)) for spec in args.agent]
    rows = []
    for item in case["record"].get("trace", []):
        if item.get("player") != args.player:
            continue
        obs = item["observation"]
        row = {
            "step": item["step"],
            "context": item.get("select_context"),
            "original_action": item.get("action"),
            "agents": {},
        }
        for spec, agent in agents:
            action = agent(obs)
            row["agents"][spec] = {"action": action, "validity": validate(obs, action)}
        rows.append(row)
        if len(rows) >= args.limit:
            break
    print(json.dumps({"case_id": case.get("case_id"), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

