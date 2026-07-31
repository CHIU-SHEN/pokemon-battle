from __future__ import annotations


def passing_metrics() -> dict:
    return {
        "illegal_actions": 0,
        "recent_history_win_rate": 0.57,
        "weak_baseline_win_rate": 0.75,
        "candidate_cross_win_rate": 0.54,
        "best_cross_win_rate": 0.55,
        "candidate_p95_seconds": 0.0020,
        "best_p95_seconds": 0.0020,
    }


def test_regression_gate_accepts_all_thresholds() -> None:
    from src.rl.selfplay_regression import regression_decision

    result = regression_decision(passing_metrics())

    assert result["passed"] is True
    assert result["failures"] == []


def test_regression_gate_rejects_weak_baseline_and_cross_degradation() -> None:
    from src.rl.selfplay_regression import regression_decision

    metrics = passing_metrics()
    metrics["weak_baseline_win_rate"] = 0.69
    metrics["candidate_cross_win_rate"] = 0.52
    result = regression_decision(metrics)

    assert result["passed"] is False
    assert "weak_baseline_below_70_percent" in result["failures"]
    assert "cross_top2_degradation_over_2pp" in result["failures"]


def test_regression_gate_rejects_history_latency_and_illegal_actions() -> None:
    from src.rl.selfplay_regression import regression_decision

    metrics = passing_metrics()
    metrics.update(
        {
            "recent_history_win_rate": 0.54,
            "candidate_p95_seconds": 0.0026,
            "illegal_actions": 1,
        }
    )
    result = regression_decision(metrics)

    assert result["passed"] is False
    assert set(result["failures"]) == {
        "illegal_actions",
        "recent_history_below_55_percent",
        "latency_over_1_25x",
    }
