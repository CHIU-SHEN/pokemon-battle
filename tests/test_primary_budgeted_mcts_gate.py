from __future__ import annotations

import pytest


def _report(**overrides) -> dict:
    report = {
        "wins": 220,
        "losses": 180,
        "draws": 0,
        "exceptions": 0,
        "illegal_actions": 0,
        "p95_decision_seconds": 0.035,
    }
    report.update(overrides)
    return report


def test_submission_gate_passes_exact_boundaries() -> None:
    from scripts.evaluate_primary_budgeted_mcts import submission_gate

    result = submission_gate(_report())
    assert result["status"] == "pass"
    assert result["win_rate"] == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"wins": 219, "losses": 180}, "requires_400_games"),
        ({"wins": 219, "losses": 181}, "win_rate_below_55_percent"),
        ({"p95_decision_seconds": 0.035001}, "p95_latency_above_35ms"),
        ({"exceptions": 1}, "exceptions_present"),
        ({"illegal_actions": 1}, "illegal_actions_present"),
    ],
)
def test_submission_gate_rejects_each_failed_boundary(overrides: dict, reason: str) -> None:
    from scripts.evaluate_primary_budgeted_mcts import submission_gate

    result = submission_gate(_report(**overrides))
    assert result == {
        "status": "reject",
        "reason": reason,
        "formal_submission_replacement_authorized": False,
    }


def test_deck_wrapper_preserves_action_source_diagnostics() -> None:
    from eval.run_match import with_deck

    class Agent:
        def __call__(self, obs):
            return [0]

        def action_source(self):
            return "mcts"

    wrapped = with_deck(Agent(), [1] * 60)
    assert wrapped.action_source() == "mcts"
