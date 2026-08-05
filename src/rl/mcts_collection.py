"""Pure planning and auditing helpers for resumable MCTS collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.rl.mcts_dataset import FORBIDDEN_HIDDEN_KEYS, _keys


@dataclass(frozen=True)
class ShardPlan:
    worker_index: int
    games: int
    seed: int
    iteration_id: str
    relative_root: str
    game_ids: tuple[str, ...]


def choose_worker_candidates(logical_cpus: int, override: int | None = None) -> tuple[int, ...]:
    if logical_cpus <= 0 or (override is not None and override <= 0):
        raise ValueError("CPU and worker counts must be positive")
    if override is not None:
        return (override,)
    if logical_cpus >= 48:
        return (12, 16, 20)
    if logical_cpus >= 32:
        return (8, 12)
    return (max(1, logical_cpus // 4),)


def _game_id(iteration_id: str, index: int) -> str:
    return f"{iteration_id}:primary:{index:06d}:side{index % 2}"


def plan_shards(
    total_games: int,
    workers: int,
    *,
    seed: int,
    iteration_id: str,
    completed_game_ids: set[str],
) -> tuple[ShardPlan, ...]:
    if total_games <= 0 or workers <= 0:
        raise ValueError("games and workers must be positive")
    pending = [(index, _game_id(iteration_id, index)) for index in range(total_games)]
    pending = [item for item in pending if item[1] not in completed_game_ids]
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(min(workers, max(1, len(pending))))]
    for ordinal, item in enumerate(pending):
        buckets[ordinal % len(buckets)].append(item)
    return tuple(
        ShardPlan(
            worker_index=index,
            games=len(bucket),
            seed=seed + index * 1_000_003,
            iteration_id=f"{iteration_id}-w{index:02d}",
            relative_root=f"shards/worker-{index:02d}",
            game_ids=tuple(game_id for _, game_id in bucket),
        )
        for index, bucket in enumerate(buckets)
        if bucket
    )


def _fallback_count(sources: dict[str, int]) -> int:
    return sum(int(count) for source, count in sources.items() if "fallback" in source)


def audit_collection(
    root: Path,
    identity: dict[str, str],
    *,
    require_all_splits: bool = False,
) -> dict[str, Any]:
    files = sorted(set(root.glob("**/games/game_*.json")) | set(root.glob("**/games/game.json")))
    game_ids: set[str] = set()
    totals = Counter()
    sources = Counter()
    splits = Counter()
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        hidden = _keys(document) & FORBIDDEN_HIDDEN_KEYS
        if hidden:
            raise ValueError(f"hidden belief fields cannot be serialized: {sorted(hidden)}")
        for key, expected in identity.items():
            if document.get(key) != expected:
                raise ValueError(f"{key} identity mismatch in {path}")
        game_id = str(document.get("game_id", ""))
        if not game_id or game_id in game_ids:
            raise ValueError(f"duplicate game_id: {game_id!r}")
        game_ids.add(game_id)
        totals["games"] += 1
        totals["exceptions"] += len(document.get("exceptions") or [])
        totals["illegal_actions"] += sum(int(value) for value in document.get("illegal_actions") or [])
        sources.update({str(key): int(value) for key, value in (document.get("action_sources") or {}).items()})
        for sample in document.get("samples") or []:
            totals["samples"] += 1
            totals["nodes"] += int(sample.get("simulations", 0))
            splits[str(sample.get("split"))] += 1
    fallbacks = _fallback_count(dict(sources))
    decisions = sum(sources.values())
    if totals["exceptions"] or totals["illegal_actions"] or fallbacks:
        raise ValueError("collection safety gate failed: exception, illegal action, or fallback observed")
    if require_all_splits and not all(splits[name] for name in ("train", "valid", "test")):
        raise ValueError("train, valid, and test splits are required")
    return {
        "games": totals["games"],
        "samples": totals["samples"],
        "nodes": totals["nodes"],
        "exceptions": totals["exceptions"],
        "illegal_actions": totals["illegal_actions"],
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / max(1, decisions),
        "action_sources": dict(sorted(sources.items())),
        "splits": {name: splits[name] for name in ("train", "valid", "test")},
        "game_ids": sorted(game_ids),
    }
