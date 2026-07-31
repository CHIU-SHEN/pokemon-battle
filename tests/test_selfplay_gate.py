from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("wins", "losses", "draws", "expected"),
    [
        (580, 420, 0, "promote_ready"),
        (520, 480, 0, "reject"),
        (550, 450, 0, "continue"),
    ],
)
def test_initial_thousand_game_gate(
    wins: int,
    losses: int,
    draws: int,
    expected: str,
) -> None:
    from src.rl.selfplay_gate import gate_decision

    assert gate_decision(wins, losses, draws).status == expected


def test_final_gate_requires_point_and_wilson_thresholds() -> None:
    from src.rl.selfplay_gate import gate_decision

    accepted = gate_decision(1680, 1320, 0)
    rejected = gate_decision(1590, 1410, 0)

    assert accepted.status == "promote_ready"
    assert accepted.wilson_low > 0.52
    assert rejected.status == "reject"


def test_draws_are_reported_but_not_used_in_win_rate() -> None:
    from src.rl.selfplay_gate import gate_decision

    decision = gate_decision(580, 420, 50)

    assert decision.games == 1050
    assert decision.non_draw_games == 1000
    assert decision.win_rate == pytest.approx(0.58)


def test_gate_rejects_invalid_counts() -> None:
    from src.rl.selfplay_gate import gate_decision

    with pytest.raises(ValueError, match="non-negative"):
        gate_decision(-1, 1, 0)
    with pytest.raises(ValueError, match="non-draw"):
        gate_decision(0, 0, 10)
