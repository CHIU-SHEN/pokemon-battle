"""Joint-action candidate generation for bounded root search."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from .fallback import is_legal_action, safe_action
from .parser import ParsedState
from .rules import choose_action, score_options


@dataclass(frozen=True)
class ActionCandidate:
    action: tuple[int, ...]
    source: str
    prior: float

    def as_list(self) -> list[int]:
        return list(self.action)


class ActionGenerator:
    def __init__(self, seed: int = 20260706) -> None:
        self.rng = random.Random(seed)

    def generate(self, parsed: ParsedState, k: int = 8) -> list[ActionCandidate]:
        select = parsed.select
        if select is None:
            return []
        seen: set[tuple[int, ...]] = set()
        candidates: list[ActionCandidate] = []

        def add(action: list[int], source: str, prior: float) -> None:
            key = tuple(action)
            if key not in seen and is_legal_action(select, action):
                seen.add(key)
                candidates.append(ActionCandidate(key, source, prior))

        add(choose_action(parsed), "v0", 1.0)
        add(safe_action(select, parsed, prefer_empty=False), "fallback", 0.2)
        if select.min_count == 0:
            add([], "empty", 0.05)

        scores = score_options(parsed)
        ranked = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)
        for rank, idx in enumerate(ranked[: max(k, select.max_count + 2)]):
            if select.min_count <= 1 <= select.max_count:
                add([idx], f"top1_{rank}", 0.8 - rank * 0.05)

        if select.max_count > 1 and ranked:
            for size in range(max(1, select.min_count), min(select.max_count, 3) + 1):
                add(ranked[:size], f"top{size}", 0.55)
            for combo in itertools.combinations(ranked[: min(6, len(ranked))], max(1, select.min_count)):
                add(list(combo), "combo", 0.35)
                if len(candidates) >= k * 2:
                    break

        attempts = 0
        while len(candidates) < k and attempts < k * 8 and select.options:
            attempts += 1
            size = self.rng.randint(select.min_count, select.max_count)
            action = [] if size == 0 else self.rng.sample(range(len(select.options)), min(size, len(select.options)))
            add(action, "random", 0.1)

        candidates.sort(key=lambda c: c.prior, reverse=True)
        return candidates[:k]

