"""Quality gate for direct MCTS teacher Arena evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def teacher_gate_decision(
    report: Mapping[str, Any],
    *,
    minimum_games: int = 400,
    minimum_win_rate: float = 0.58,
) -> dict[str, Any]:
    if minimum_games <= 0:
        raise ValueError("minimum_games must be positive")
    if not 0.0 <= minimum_win_rate <= 1.0:
        raise ValueError("minimum_win_rate must be between zero and one")
    counts = {
        name: int(report.get(name, 0))
        for name in ("wins", "losses", "draws", "exceptions", "illegal_actions")
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("evaluation counts must be nonnegative")
    fallback_rate = float(report.get("fallback_rate", 0.0))
    if fallback_rate < 0.0:
        raise ValueError("fallback_rate must be nonnegative")
    games = counts["wins"] + counts["losses"] + counts["draws"]
    decisive_games = counts["wins"] + counts["losses"]
    win_rate = counts["wins"] / decisive_games if decisive_games else 0.0
    unsafe = (
        counts["exceptions"] > 0
        or counts["illegal_actions"] > 0
        or fallback_rate > 0.0
    )
    if games < minimum_games:
        status, reason = "fail", "teacher_evaluation_incomplete"
    elif unsafe:
        status, reason = "fail", "teacher_safety_failure"
    elif win_rate < minimum_win_rate:
        status, reason = "fail", "teacher_win_rate_below_threshold"
    else:
        status, reason = "pass", "teacher_strong_enough"
    return {
        "schema_version": "mcts_teacher_quality_gate_v1",
        "status": status,
        "reason": reason,
        "minimum_games": minimum_games,
        "minimum_win_rate": minimum_win_rate,
        "games": games,
        "decisive_games": decisive_games,
        "win_rate": win_rate,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "draws": counts["draws"],
        "exceptions": counts["exceptions"],
        "illegal_actions": counts["illegal_actions"],
        "fallback_rate": fallback_rate,
    }
