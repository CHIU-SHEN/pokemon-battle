#!/usr/bin/env python3
"""Randomized legality checks for the M1 fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from agent.fallback import is_legal_action, safe_action  # noqa: E402


@dataclass
class DummyOption:
    type: int
    cardId: int | None = None


@dataclass
class DummySelect:
    minCount: int
    maxCount: int
    option: list[DummyOption]


def main() -> int:
    rng = random.Random(20260706)
    card_pool = [None, 3, 721, 722, 723, 1145, 1158, 1205, 1227, 1235, 9999]
    for i in range(10000):
        option_count = rng.randint(0, 12)
        max_count = rng.randint(0, option_count) if option_count else 0
        min_count = rng.randint(0, max_count) if max_count else 0
        select = DummySelect(
            minCount=min_count,
            maxCount=max_count,
            option=[
                DummyOption(type=rng.randint(0, 16), cardId=rng.choice(card_pool))
                for _ in range(option_count)
            ],
        )
        action = safe_action(select, prefer_empty=bool(rng.getrandbits(1)))
        if not is_legal_action(select, action):
            raise AssertionError(f"illegal fallback at case {i}: {select} -> {action}")
    print("OK: 10000 fallback samples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

