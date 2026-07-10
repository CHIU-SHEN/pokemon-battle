#!/usr/bin/env python3
"""Minimal M0 regression checks for saved observations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "observations.json"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from cg.api import to_observation_class  # noqa: E402


def load_submission_agent():
    path = SUBMISSION_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("submission_main", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    os.chdir(SUBMISSION_DIR)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module.agent


def validate_action(obs_dict: dict, action: list[int]) -> None:
    obs = to_observation_class(obs_dict)
    select = obs.select
    assert isinstance(action, list), "agent must return list[int]"
    assert all(isinstance(x, int) for x in action), "agent action contains non-int values"
    assert select is not None, "fixtures should contain selection observations"
    assert select.minCount <= len(action) <= select.maxCount, (
        f"length {len(action)} outside [{select.minCount}, {select.maxCount}]"
    )
    assert len(action) == len(set(action)), "duplicate option indices"
    assert all(0 <= x < len(select.option) for x in action), "option index out of range"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=str(FIXTURE_PATH))
    args = parser.parse_args()

    fixture_path = Path(args.fixtures)
    with fixture_path.open("r", encoding="utf-8") as f:
        observations = json.load(f)
    assert 20 <= len(observations) <= 50, "M0 expects 20-50 saved observations"

    agent = load_submission_agent()
    deck = agent(None)
    assert isinstance(deck, list), "agent(None) must return deck list"
    assert len(deck) == 60, "deck must contain 60 cards"
    assert all(isinstance(x, int) for x in deck), "deck must contain only integers"

    for index, obs in enumerate(observations):
        try:
            validate_action(obs, agent(obs))
        except Exception as exc:
            raise AssertionError(f"fixture {index} failed: {exc}") from exc

    print(f"OK: {len(observations)} fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
