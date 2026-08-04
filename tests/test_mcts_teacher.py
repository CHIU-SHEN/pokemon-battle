from __future__ import annotations

import math

import pytest
import torch


def _window(
    relative_update: float,
    policy_loss: float,
    *,
    value_loss: float = 0.5,
    reference_kl: float = 0.01,
) -> dict[str, float]:
    return {
        "relative_update_ema": relative_update,
        "holdout_policy_loss": policy_loss,
        "holdout_value_loss": value_loss,
        "holdout_reference_kl": reference_kl,
    }


def test_relative_parameter_update_matches_hand_calculation() -> None:
    from src.rl.mcts_teacher import relative_parameter_update, snapshot_parameters

    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    before = snapshot_parameters([parameter])
    parameter.data.add_(torch.tensor([0.3, 0.4]))

    assert relative_parameter_update(before, [parameter]) == pytest.approx(0.1)


def test_gradient_norm_matches_hand_calculation() -> None:
    from src.rl.mcts_teacher import gradient_norm

    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    assert gradient_norm([parameter]) == pytest.approx(5.0)


def test_three_flat_windows_converge() -> None:
    from src.rl.mcts_teacher import TeacherConvergenceConfig, evaluate_teacher_stop

    history = [
        _window(1e-6, 1.0, value_loss=0.5),
        _window(1e-6, 0.999, value_loss=0.501),
        _window(1e-6, 0.9985, value_loss=0.502),
        _window(1e-6, 0.998, value_loss=0.503),
    ]

    decision = evaluate_teacher_stop(history, TeacherConvergenceConfig())

    assert decision.stop is True
    assert decision.converged is True
    assert decision.reason == "converged"


def test_small_updates_do_not_stop_while_holdout_improves() -> None:
    from src.rl.mcts_teacher import TeacherConvergenceConfig, evaluate_teacher_stop

    history = [_window(1e-6, value) for value in (1.0, 0.9, 0.8, 0.7)]

    assert evaluate_teacher_stop(history, TeacherConvergenceConfig()).stop is False


def test_two_worsening_policy_windows_stop_as_unsafe() -> None:
    from src.rl.mcts_teacher import TeacherConvergenceConfig, evaluate_teacher_stop

    history = [_window(1e-4, value) for value in (1.0, 1.1, 1.2)]
    decision = evaluate_teacher_stop(history, TeacherConvergenceConfig())

    assert decision.stop is True
    assert decision.unsafe is True
    assert decision.reason == "holdout_policy_worsened"


@pytest.mark.parametrize(
    ("history", "reason"),
    [
        ([_window(1e-4, 1.0, reference_kl=0.031)], "reference_kl_limit"),
        ([_window(math.nan, 1.0)], "non_finite"),
    ],
)
def test_safety_limits_stop_training(history, reason) -> None:
    from src.rl.mcts_teacher import TeacherConvergenceConfig, evaluate_teacher_stop

    decision = evaluate_teacher_stop(history, TeacherConvergenceConfig(reference_kl_max=0.03))

    assert decision.stop is True
    assert decision.unsafe is True
    assert decision.reason == reason


def test_wall_time_stop_is_not_convergence() -> None:
    from src.rl.mcts_teacher import TeacherConvergenceConfig, evaluate_teacher_stop

    decision = evaluate_teacher_stop(
        [_window(1e-3, 1.0)],
        TeacherConvergenceConfig(max_wall_seconds=10.0),
        elapsed_seconds=10.0,
    )

    assert decision.stop is True
    assert decision.converged is False
    assert decision.unsafe is False
    assert decision.reason == "wall_time_limit"
