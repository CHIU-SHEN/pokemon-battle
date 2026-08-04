"""Pilot-only gates for MCTS search and distillation uplift."""

from __future__ import annotations

from dataclasses import dataclass

from src.rl.selfplay_gate import wilson_interval


@dataclass(frozen=True)
class MCTSGateDecision:
    status: str
    reason: str
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float
    wilson_low: float
    wilson_high: float


def mcts_gate_decision(
    wins: int,
    losses: int,
    draws: int,
    *,
    kind: str,
    exceptions: int = 0,
    illegal_actions: int = 0,
    fallback_rate: float = 0.0,
) -> MCTSGateDecision:
    if kind not in {"search", "candidate"}:
        raise ValueError("kind must be search or candidate")
    games = wins + losses + draws
    if wins + losses <= 0:
        raise ValueError("at least one non-draw game is required")
    rate = wins / (wins + losses)
    low, high = wilson_interval(wins, losses)
    if exceptions or illegal_actions or fallback_rate >= 0.05:
        status, reason = "reject", "safety_failure"
    elif kind == "search":
        status, reason = (
            ("pass", "search_rate_at_least_55_percent")
            if games >= 400 and rate >= 0.55
            else ("reject", "search_uplift_not_proven")
        )
    elif games < 400:
        status, reason = "continue", "candidate_smoke_only"
    elif games < 1000:
        if rate >= 0.53:
            status, reason = "pass", "candidate_rate_at_least_53_percent"
        else:
            status, reason = "continue", "candidate_gray_zone"
    else:
        status, reason = (
            ("pass", "candidate_final_rate_at_least_55_percent")
            if rate >= 0.55
            else ("reject", "candidate_final_rate_below_55_percent")
        )
    return MCTSGateDecision(
        status=status,
        reason=reason,
        wins=wins,
        losses=losses,
        draws=draws,
        games=games,
        win_rate=rate,
        wilson_low=low,
        wilson_high=high,
    )
