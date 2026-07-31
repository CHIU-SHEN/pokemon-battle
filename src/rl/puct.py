"""Pure PUCT statistics and probability helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


Action = tuple[int, ...]


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += float(value)


@dataclass
class NodeStats:
    edges: dict[Action, EdgeStats] = field(default_factory=dict)

    @property
    def visits(self) -> int:
        return sum(edge.visits for edge in self.edges.values())


def puct_score(edge: EdgeStats, *, parent_visits: int, c_puct: float) -> float:
    if edge.prior < 0.0 or edge.visits < 0 or parent_visits < 0:
        raise ValueError("PUCT counts and priors must be non-negative")
    if not math.isfinite(c_puct) or c_puct < 0.0:
        raise ValueError("c_puct must be finite and non-negative")
    exploration = c_puct * edge.prior * math.sqrt(max(1, parent_visits)) / (1 + edge.visits)
    return edge.mean_value + exploration


def select_puct_action(edges: Mapping[Action, EdgeStats], *, c_puct: float) -> Action:
    if not edges:
        raise ValueError("at least one PUCT edge is required")
    parent_visits = sum(edge.visits for edge in edges.values())
    return max(
        sorted(edges),
        key=lambda action: puct_score(
            edges[action],
            parent_visits=parent_visits,
            c_puct=c_puct,
        ),
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def joint_action_priors(
    *,
    option_logits: Sequence[float],
    actions: Sequence[Action],
) -> dict[Action, float]:
    if not actions:
        raise ValueError("at least one joint action is required")
    logits = np.asarray(option_logits, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("option_logits must be one finite vector")
    log_probs = logits - math.log(float(np.exp(logits - logits.max()).sum())) - logits.max()
    scores = []
    normalized_actions: list[Action] = []
    for raw_action in actions:
        action = tuple(int(index) for index in raw_action)
        if not action or len(set(action)) != len(action):
            raise ValueError("joint actions must be non-empty and unique")
        if min(action) < 0 or max(action) >= len(log_probs):
            raise ValueError("joint action index outside option logits")
        normalized_actions.append(action)
        scores.append(float(log_probs[list(action)].sum()) / math.sqrt(len(action)))
    probabilities = _softmax(np.asarray(scores, dtype=np.float64))
    return {action: float(probability) for action, probability in zip(normalized_actions, probabilities)}


def visit_distribution(
    edges: Mapping[Action, EdgeStats],
    *,
    temperature: float,
) -> dict[Action, float]:
    if not edges:
        raise ValueError("at least one visited edge is required")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    actions = sorted(edges)
    visits = np.asarray([edges[action].visits for action in actions], dtype=np.float64)
    if temperature == 0.0:
        winner = int(np.argmax(visits))
        return {action: 1.0 if index == winner else 0.0 for index, action in enumerate(actions)}
    if visits.sum() == 0.0:
        weights = np.asarray([edges[action].prior for action in actions], dtype=np.float64)
    else:
        weights = np.power(visits, 1.0 / temperature)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    probabilities = weights / weights.sum()
    return {action: float(probability) for action, probability in zip(actions, probabilities)}


def mix_dirichlet_noise(
    priors: Mapping[Action, float],
    *,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[Action, float]:
    if not priors:
        raise ValueError("at least one prior is required")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    actions = sorted(priors)
    base = np.asarray([priors[action] for action in actions], dtype=np.float64)
    if (base < 0.0).any() or base.sum() <= 0.0:
        raise ValueError("priors must be non-negative with positive mass")
    base /= base.sum()
    noise = rng.dirichlet(np.full(len(actions), alpha, dtype=np.float64))
    mixed = (1.0 - epsilon) * base + epsilon * noise
    mixed /= mixed.sum()
    return {action: float(probability) for action, probability in zip(actions, mixed)}
