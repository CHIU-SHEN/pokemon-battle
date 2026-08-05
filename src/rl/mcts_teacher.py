"""Convergence and parameter-update diagnostics for MCTS teacher training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TeacherConvergenceConfig:
    relative_update_max: float = 1e-5
    policy_improvement_max: float = 0.002
    value_worsening_max: float = 0.01
    reference_kl_max: float = 0.03
    patience: int = 3
    policy_worsening_patience: int = 2
    max_wall_seconds: float = 21_600.0


@dataclass(frozen=True)
class TeacherStopDecision:
    stop: bool
    reason: str | None = None
    converged: bool = False
    unsafe: bool = False


def adapt_kl_coefficient(
    holdout_kl: float,
    current: float,
    *,
    hard_limit: float = 0.03,
    minimum: float = 0.01,
    maximum: float = 1.0,
) -> float:
    """Return the next epoch's KL coefficient or reject an unsafe epoch."""
    if not math.isfinite(holdout_kl) or not math.isfinite(current) or current <= 0:
        raise ValueError("KL values must be finite and the coefficient positive")
    if holdout_kl > hard_limit:
        raise ValueError("holdout KL exceeded the hard limit")
    if holdout_kl < 0.015:
        return max(minimum, current * 0.8)
    if holdout_kl <= 0.025:
        return current
    return min(maximum, current * 2.0)


def _parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return list(parameters)


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in _parameters(parameters):
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().double().norm(2).item())
        squared += value * value
    return math.sqrt(squared)


def snapshot_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in _parameters(parameters)]


def relative_parameter_update(
    before: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> float:
    current = _parameters(parameters)
    if len(before) != len(current):
        raise ValueError("parameter snapshot length mismatch")
    delta_squared = 0.0
    before_squared = 0.0
    for old, parameter in zip(before, current):
        if old.shape != parameter.shape:
            raise ValueError("parameter snapshot shape mismatch")
        old_value = old.detach().double()
        new_value = parameter.detach().double()
        delta_squared += float((new_value - old_value).square().sum().item())
        before_squared += float(old_value.square().sum().item())
    result = math.sqrt(delta_squared) / max(math.sqrt(before_squared), 1e-12)
    if not math.isfinite(result):
        raise ValueError("non-finite relative parameter update")
    return result


def ema(previous: float | None, value: float, decay: float = 0.9) -> float:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    return float(value) if previous is None else decay * float(previous) + (1.0 - decay) * float(value)


def _relative_improvement(previous: float, current: float) -> float:
    return (previous - current) / max(abs(previous), 1e-12)


def _relative_worsening(previous: float, current: float) -> float:
    return (current - previous) / max(abs(previous), 1e-12)


def evaluate_teacher_stop(
    history: Sequence[dict[str, float]],
    config: TeacherConvergenceConfig,
    *,
    elapsed_seconds: float = 0.0,
) -> TeacherStopDecision:
    required = {
        "relative_update_ema",
        "holdout_policy_loss",
        "holdout_value_loss",
        "holdout_reference_kl",
    }
    for window in history:
        values = [float(window[key]) for key in required]
        if not all(math.isfinite(value) for value in values):
            return TeacherStopDecision(True, "non_finite", unsafe=True)
        if float(window["holdout_reference_kl"]) > config.reference_kl_max:
            return TeacherStopDecision(True, "reference_kl_limit", unsafe=True)

    worsening = 0
    for previous, current in zip(history, history[1:]):
        if float(current["holdout_policy_loss"]) > float(previous["holdout_policy_loss"]):
            worsening += 1
        else:
            worsening = 0
        if worsening >= config.policy_worsening_patience:
            return TeacherStopDecision(True, "holdout_policy_worsened", unsafe=True)

    if len(history) >= config.patience + 1:
        comparisons = list(zip(history[-config.patience - 1:-1], history[-config.patience:]))
        plateau = all(
            float(current["relative_update_ema"]) < config.relative_update_max
            and _relative_improvement(
                float(previous["holdout_policy_loss"]),
                float(current["holdout_policy_loss"]),
            ) < config.policy_improvement_max
            and _relative_worsening(
                float(previous["holdout_value_loss"]),
                float(current["holdout_value_loss"]),
            ) <= config.value_worsening_max
            for previous, current in comparisons
        )
        if plateau:
            return TeacherStopDecision(True, "converged", converged=True)

    if config.max_wall_seconds > 0.0 and elapsed_seconds >= config.max_wall_seconds:
        return TeacherStopDecision(True, "wall_time_limit")
    return TeacherStopDecision(False)
