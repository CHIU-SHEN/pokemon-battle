"""Deterministic opponent-pool scheduling for gated Top2 self-play."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class OpponentSpec:
    kind: str
    name: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    branch: str | None
    deck_id: str | None
    checkpoint_kind: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_spec(kind: str, item: dict[str, Any]) -> OpponentSpec:
    path = Path(item["path"])
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise ValueError(f"opponent checkpoint hash mismatch: {path}")
    return OpponentSpec(
        kind=kind,
        name=Path(path).stem,
        checkpoint=str(path),
        checkpoint_sha256=actual,
        branch=item.get("branch"),
        deck_id=item.get("deck_id"),
        checkpoint_kind=item.get("checkpoint_kind", "ppo"),
    )


def _quota(games: int, fraction: float) -> int:
    return int(games * fraction)


def build_opponent_schedule(
    *,
    best: dict[str, Any],
    history: list[dict[str, Any]],
    games: int,
    seed: int,
    baselines: Iterable[str] = ("random", "first-min"),
) -> list[OpponentSpec]:
    if games <= 0:
        raise ValueError("games must be positive")
    baseline_names = tuple(baselines)
    if not baseline_names:
        raise ValueError("at least one baseline is required")
    history_games = _quota(games, 0.30) if history else 0
    baseline_games = _quota(games, 0.20)
    best_games = games - history_games - baseline_games
    schedule = [_checkpoint_spec("best", best) for _ in range(best_games)]
    if history_games:
        recent_first = list(reversed(history))
        for index in range(history_games):
            schedule.append(_checkpoint_spec("history", recent_first[index % len(recent_first)]))
    for index in range(baseline_games):
        name = baseline_names[index % len(baseline_names)]
        schedule.append(
            OpponentSpec(
                kind="baseline",
                name=name,
                checkpoint=None,
                checkpoint_sha256=None,
                branch=None,
                deck_id=None,
                checkpoint_kind=None,
            )
        )
    random.Random(seed).shuffle(schedule)
    return schedule


def validate_checkpoint_identity(
    path: Path,
    *,
    expected_candidate_id: str,
    expected_deck_id: str,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "top2_ppo_checkpoint_v1":
        raise ValueError("unsupported PPO checkpoint schema")
    if checkpoint.get("deck_id") != expected_deck_id:
        raise ValueError(
            f"checkpoint deck_id mismatch: {checkpoint.get('deck_id')} != {expected_deck_id}"
        )
    if checkpoint.get("candidate_id") != expected_candidate_id:
        raise ValueError(
            "checkpoint candidate_id mismatch: "
            f"{checkpoint.get('candidate_id')} != {expected_candidate_id}"
        )
    return checkpoint


def build_game_identity(
    *,
    iteration_id: str,
    game_index: int,
    learner: dict[str, Any],
    opponent: dict[str, Any],
) -> dict[str, Any]:
    if not iteration_id:
        raise ValueError("iteration_id is required")
    return {
        "iteration_id": iteration_id,
        "game_index": int(game_index),
        "learner_sha256": learner["sha256"],
        "opponent_sha256": opponent["sha256"],
    }
