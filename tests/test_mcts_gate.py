from __future__ import annotations


def test_search_gate_requires_55_percent() -> None:
    from src.rl.mcts_gate import mcts_gate_decision

    passed = mcts_gate_decision(220, 180, 0, kind="search")
    failed = mcts_gate_decision(219, 181, 0, kind="search")

    assert passed.status == "pass"
    assert failed.status == "reject"


def test_candidate_gate_uses_53_percent_at_400_and_55_percent_at_1000() -> None:
    from src.rl.mcts_gate import mcts_gate_decision

    assert mcts_gate_decision(212, 188, 0, kind="candidate").status == "pass"
    assert mcts_gate_decision(210, 190, 0, kind="candidate").status == "continue"
    assert mcts_gate_decision(540, 460, 0, kind="candidate").status == "reject"
    assert mcts_gate_decision(550, 450, 0, kind="candidate").status == "pass"


def test_safety_failure_always_rejects() -> None:
    from src.rl.mcts_gate import mcts_gate_decision

    decision = mcts_gate_decision(
        300,
        100,
        0,
        kind="search",
        exceptions=1,
    )

    assert decision.status == "reject"
    assert decision.reason == "safety_failure"
