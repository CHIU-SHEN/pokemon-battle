"""Hard regression gates applied after candidate-vs-best promotion readiness."""

from __future__ import annotations

from typing import Any


def regression_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if int(metrics.get("illegal_actions", 0)) != 0:
        failures.append("illegal_actions")
    history_rate = metrics.get("recent_history_win_rate")
    if history_rate is not None and float(history_rate) < 0.55:
        failures.append("recent_history_below_55_percent")
    weak_rate = metrics.get("weak_baseline_win_rate")
    if weak_rate is not None and float(weak_rate) < 0.70:
        failures.append("weak_baseline_below_70_percent")
    candidate_cross = metrics.get("candidate_cross_win_rate")
    best_cross = metrics.get("best_cross_win_rate")
    if candidate_cross is not None and best_cross is not None:
        if float(candidate_cross) < float(best_cross) - 0.02:
            failures.append("cross_top2_degradation_over_2pp")
    candidate_latency = metrics.get("candidate_p95_seconds")
    best_latency = metrics.get("best_p95_seconds")
    if candidate_latency is not None and best_latency is not None:
        if float(candidate_latency) > float(best_latency) * 1.25:
            failures.append("latency_over_1_25x")
    return {
        **metrics,
        "passed": not failures,
        "failures": failures,
    }
