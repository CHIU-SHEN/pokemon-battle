"""Pure policy helpers for the time-bounded Top2 local PPO pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PilotBudget:
    name: str
    rollout_games_per_branch: int
    arena_games: int
    trials: tuple[str, ...]
    epoch_cap: int | None = None


def choose_budget_tier(predicted_seconds: float) -> PilotBudget:
    if predicted_seconds <= 5400.0:
        return PilotBudget("full", 100, 200, ("conservative", "baseline", "exploratory"))
    if predicted_seconds <= 7200.0:
        return PilotBudget("reduced", 100, 100, ("conservative", "baseline", "exploratory"))
    return PilotBudget("minimal", 100, 0, ("conservative", "baseline"), epoch_cap=2)


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0 or wins < 0 or wins > games:
        raise ValueError("Wilson interval requires 0 <= wins <= games and games > 0")
    rate = wins / games
    z2 = z * z
    denominator = 1.0 + z2 / games
    center = (rate + z2 / (2.0 * games)) / denominator
    margin = z * math.sqrt((rate * (1.0 - rate) + z2 / (4.0 * games)) / games) / denominator
    return center - margin, center + margin


def select_preliminary_trial(trials: list[dict]) -> dict:
    eligible = []
    for source in trials:
        if not source.get("eligible"):
            continue
        item = dict(source)
        item["arena_wilson_95"] = list(wilson_interval(int(item["arena_wins"]), int(item["arena_games"])))
        eligible.append(item)
    if not eligible:
        return {"status": "no_eligible_trial", "selected": None, "eligible": []}
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item["arena_wilson_95"][0]),
            -int(item["arena_wins"]) / int(item["arena_games"]),
            str(item["name"]),
        ),
    )
    if len(ranked) == 1 or ranked[0]["arena_wilson_95"][0] > ranked[1]["arena_wilson_95"][1]:
        return {"status": "preliminary_separated", "selected": ranked[0]["name"], "eligible": ranked}
    by_name = {item["name"]: item for item in eligible}
    selected = next((name for name in ("conservative", "baseline", "exploratory") if name in by_name), ranked[0]["name"])
    return {"status": "preliminary_intervals_overlap", "selected": selected, "eligible": ranked}
