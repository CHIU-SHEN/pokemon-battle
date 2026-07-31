"""Statistical promotion gates for Top2 self-play candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class GateDecision:
    status: str
    reason: str
    wins: int
    losses: int
    draws: int
    games: int
    non_draw_games: int
    win_rate: float
    wilson_low: float
    wilson_high: float
    next_non_draw_target: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(
    wins: int,
    losses: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if wins < 0 or losses < 0:
        raise ValueError("counts must be non-negative")
    total = wins + losses
    if total == 0:
        raise ValueError("at least one non-draw game is required")
    proportion = wins / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = proportion + z2 / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def gate_decision(
    wins: int,
    losses: int,
    draws: int,
    *,
    initial_games: int = 1000,
    games_cap: int = 3000,
) -> GateDecision:
    if min(wins, losses, draws) < 0:
        raise ValueError("counts must be non-negative")
    non_draw_games = wins + losses
    if non_draw_games == 0:
        raise ValueError("at least one non-draw game is required")
    if initial_games <= 0 or games_cap < initial_games:
        raise ValueError("invalid gate game limits")
    rate = wins / non_draw_games
    low, high = wilson_interval(wins, losses)

    if non_draw_games < initial_games:
        status, reason, target = "continue", "initial_sample_incomplete", initial_games
    elif non_draw_games == initial_games:
        if rate >= 0.58:
            status, reason, target = "promote_ready", "initial_rate_at_least_58_percent", None
        elif rate <= 0.52:
            status, reason, target = "reject", "initial_rate_at_most_52_percent", None
        else:
            status, reason, target = "continue", "initial_rate_in_gray_zone", games_cap
    elif non_draw_games < games_cap:
        status, reason, target = "continue", "gray_zone_sample_incomplete", games_cap
    elif rate >= 0.55 and low > 0.52:
        status, reason, target = "promote_ready", "final_rate_and_wilson_gate_passed", None
    else:
        status, reason, target = "reject", "final_rate_or_wilson_gate_failed", None

    return GateDecision(
        status=status,
        reason=reason,
        wins=wins,
        losses=losses,
        draws=draws,
        games=wins + losses + draws,
        non_draw_games=non_draw_games,
        win_rate=rate,
        wilson_low=low,
        wilson_high=high,
        next_non_draw_target=target,
    )
