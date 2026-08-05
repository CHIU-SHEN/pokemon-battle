from __future__ import annotations

import json
from pathlib import Path

import pytest


def _report(
    *,
    wins: int = 232,
    losses: int = 168,
    draws: int = 0,
    exceptions: int = 0,
    illegal_actions: int = 0,
    fallback_rate: float = 0.0,
) -> dict:
    return {
        "schema_version": "top2_mcts_eval_v1",
        "branch": "primary",
        "kind": "search",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "exceptions": exceptions,
        "illegal_actions": illegal_actions,
        "fallback_rate": fallback_rate,
    }


@pytest.mark.parametrize(
    ("report", "status", "reason"),
    [
        (_report(), "pass", "teacher_strong_enough"),
        (_report(wins=231, losses=169), "fail", "teacher_win_rate_below_threshold"),
        (_report(wins=200, losses=199), "fail", "teacher_evaluation_incomplete"),
        (_report(exceptions=1), "fail", "teacher_safety_failure"),
        (_report(illegal_actions=1), "fail", "teacher_safety_failure"),
        (_report(fallback_rate=0.001), "fail", "teacher_safety_failure"),
    ],
)
def test_teacher_gate_enforces_strength_completeness_and_safety(
    report: dict, status: str, reason: str
) -> None:
    from src.rl.mcts_teacher_gate import teacher_gate_decision

    decision = teacher_gate_decision(report)

    assert decision["status"] == status
    assert decision["reason"] == reason
    assert decision["minimum_games"] == 400
    assert decision["minimum_win_rate"] == 0.58


def test_teacher_gate_rejects_invalid_counts() -> None:
    from src.rl.mcts_teacher_gate import teacher_gate_decision

    with pytest.raises(ValueError, match="nonnegative"):
        teacher_gate_decision(_report(wins=-1))


def test_gate_cli_writes_decision_and_returns_two_for_weak_teacher(
    tmp_path: Path,
) -> None:
    from scripts.gate_mcts_teacher import main

    source = tmp_path / "teacher-eval.json"
    output = tmp_path / "teacher-gate.json"
    source.write_text(json.dumps(_report(wins=231, losses=169)), encoding="utf-8")

    exit_code = main([str(source), "--output", str(output)])

    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["reason"] == (
        "teacher_win_rate_below_threshold"
    )
