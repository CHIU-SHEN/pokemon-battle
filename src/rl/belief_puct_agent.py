"""Neural belief-PUCT search independent of a particular model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.rl.puct import (
    Action,
    EdgeStats,
    joint_action_priors,
    mix_dirichlet_noise,
    select_puct_action,
    visit_distribution,
)


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    particles: int = 3
    max_depth: int = 8
    c_puct: float = 1.5
    root_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    seed: int = 20260731

    def __post_init__(self) -> None:
        if min(self.simulations, self.particles, self.max_depth) <= 0:
            raise ValueError("search budgets must be positive")


@dataclass(frozen=True)
class NodeEvaluation:
    actions: tuple[Action, ...]
    option_logits: tuple[float, ...]
    value: float
    player: int
    result: int = -1


@dataclass(frozen=True)
class SearchDecision:
    action: Action | None
    visit_counts: dict[Action, int]
    policy_target: dict[Action, float]
    root_value: float
    simulations: int
    particles_requested: int
    particles_valid: int
    max_depth_reached: int
    fallback_reason: str | None
    backend_report: dict[str, Any]


@dataclass
class _TreeNode:
    state_id: int
    evaluation: NodeEvaluation
    edges: dict[Action, EdgeStats]
    children: dict[Action, int] = field(default_factory=dict)


def backup_value_for_player(
    root_value: float,
    *,
    root_player: int,
    parent_player: int,
) -> float:
    return float(root_value) if parent_player == root_player else -float(root_value)


def _root_value(evaluation: NodeEvaluation, root_player: int) -> float:
    if evaluation.result != -1:
        if evaluation.result == 2:
            return 0.0
        return 1.0 if evaluation.result == root_player else -1.0
    value = max(-1.0, min(1.0, float(evaluation.value)))
    return value if evaluation.player == root_player else -value


class BeliefPUCTSearch:
    def __init__(
        self,
        *,
        backend: Any,
        evaluator: Callable[[Any], NodeEvaluation],
        config: SearchConfig | None = None,
    ) -> None:
        self.backend = backend
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _make_node(self, state_id: int, observation: Any, *, root: bool) -> _TreeNode:
        evaluation = self.evaluator(observation)
        priors = (
            joint_action_priors(
                option_logits=evaluation.option_logits,
                actions=evaluation.actions,
            )
            if evaluation.actions
            else {}
        )
        if root and self.config.root_noise and len(priors) > 1:
            priors = mix_dirichlet_noise(
                priors,
                alpha=self.config.dirichlet_alpha,
                epsilon=self.config.dirichlet_epsilon,
                rng=self.rng,
            )
        return _TreeNode(
            state_id=state_id,
            evaluation=evaluation,
            edges={action: EdgeStats(prior=prior) for action, prior in priors.items()},
        )

    def _simulate(
        self,
        *,
        root_id: int,
        nodes: dict[int, _TreeNode],
        root_player: int,
    ) -> tuple[float, int]:
        node = nodes[root_id]
        path: list[tuple[_TreeNode, Action]] = []
        depth = 0
        while node.edges and depth < self.config.max_depth:
            action = select_puct_action(node.edges, c_puct=self.config.c_puct)
            path.append((node, action))
            child_id = node.children.get(action)
            depth += 1
            if child_id is None:
                child = self.backend.step(node.state_id, action)
                child_node = self._make_node(child.state_id, child.observation, root=False)
                nodes[child.state_id] = child_node
                node.children[action] = child.state_id
                node = child_node
                break
            node = nodes[child_id]
        value = _root_value(node.evaluation, root_player)
        for parent, action in reversed(path):
            parent.edges[action].backup(
                backup_value_for_player(
                    value,
                    root_player=root_player,
                    parent_player=parent.evaluation.player,
                )
            )
        return value, depth

    def search(self, observation: Any, ledger: Any, *, temperature: float) -> SearchDecision:
        roots = []
        trees: list[dict[int, _TreeNode]] = []
        root_values = []
        max_depth_reached = 0
        try:
            roots = self.backend.begin_particles(
                observation,
                ledger,
                count=self.config.particles,
            )
            if not roots:
                return SearchDecision(
                    action=None,
                    visit_counts={},
                    policy_target={},
                    root_value=0.0,
                    simulations=0,
                    particles_requested=self.config.particles,
                    particles_valid=0,
                    max_depth_reached=0,
                    fallback_reason="no_valid_particles",
                    backend_report=self.backend.report(),
                )
            root_player = self.evaluator(roots[0].observation).player
            for root in roots:
                node = self._make_node(root.state_id, root.observation, root=True)
                trees.append({root.state_id: node})
            for simulation in range(self.config.simulations):
                particle_index = simulation % len(roots)
                value, depth = self._simulate(
                    root_id=roots[particle_index].state_id,
                    nodes=trees[particle_index],
                    root_player=root_player,
                )
                root_values.append(value)
                max_depth_reached = max(max_depth_reached, depth)
            actions = sorted(
                {
                    action
                    for tree, root in zip(trees, roots)
                    for action in tree[root.state_id].edges
                }
            )
            visits = {
                action: sum(
                    tree[root.state_id].edges.get(action, EdgeStats(0.0)).visits
                    for tree, root in zip(trees, roots)
                )
                for action in actions
            }
            aggregate = {
                action: EdgeStats(prior=0.0, visits=count)
                for action, count in visits.items()
            }
            target = visit_distribution(aggregate, temperature=temperature)
            selected = max(sorted(target), key=target.get) if target else None
            return SearchDecision(
                action=selected,
                visit_counts=visits,
                policy_target=target,
                root_value=float(np.mean(root_values)) if root_values else 0.0,
                simulations=self.config.simulations,
                particles_requested=self.config.particles,
                particles_valid=len(roots),
                max_depth_reached=max_depth_reached,
                fallback_reason=None if selected is not None else "no_legal_search_actions",
                backend_report=self.backend.report(),
            )
        finally:
            self.backend.close()


class Top2NodeEvaluator:
    """Adapt a Top2 policy/value model and joint-action generator to PUCT."""

    def __init__(self, policy: Any, *, max_candidates: int = 8) -> None:
        from agent.action_gen import ActionGenerator

        self.policy = policy
        self.generator = ActionGenerator()
        self.max_candidates = max_candidates
        self.last_public_features: dict[str, Any] = {}

    def __call__(self, observation: Any) -> NodeEvaluation:
        parsed, logits, value, features = self.policy.evaluate_policy_value(observation)
        self.last_public_features = features
        result = int(parsed.result)
        if result != -1 or parsed.select is None:
            return NodeEvaluation((), tuple(logits), value, parsed.current_player, result)
        candidates = self.generator.generate(parsed, self.max_candidates)
        actions = tuple(candidate.action for candidate in candidates)
        return NodeEvaluation(actions, tuple(logits), value, parsed.current_player, result)


class Top2BeliefPUCTAgent:
    """Agent-compatible MCTS wrapper with a safe policy fallback."""

    def __init__(
        self,
        policy: Any,
        *,
        config: SearchConfig | None = None,
        selfplay: bool = True,
    ) -> None:
        from agent.belief import BeliefSampler
        from agent.parser import GameLedger

        self.policy = policy
        self.config = config or SearchConfig(root_noise=selfplay)
        self.sampler = BeliefSampler(deck=policy.deck, seed=self.config.seed)
        self.ledger = GameLedger()
        self.last_decision: SearchDecision | None = None
        self._last_source = "initialized"
        self.searchable_decisions = 0

    def action_source(self) -> str:
        return self._last_source

    def __call__(self, obs_dict: dict | None) -> list[int]:
        if obs_dict is None or obs_dict.get("select") is None:
            self._last_source = "deck"
            return list(self.policy.deck)
        from agent.fallback import is_legal_action
        from agent.parser import parse_observation
        from cg.api import to_observation_class
        from src.rl.search_backend import SearchBackend

        obs = to_observation_class(obs_dict)
        parsed = parse_observation(obs)
        self.ledger.update(parsed)
        select = parsed.select
        if (
            select is None
            or len(select.options) <= 1
            or not getattr(obs, "search_begin_input", None)
            or (select.min_count != select.max_count and select.max_count > 2)
        ):
            self.last_decision = None
            self._last_source = "policy_fallback"
            return self.policy(obs_dict)
        evaluator = Top2NodeEvaluator(self.policy)
        backend = SearchBackend(sampler=self.sampler)
        temperature = 1.0 if self.searchable_decisions < 20 else 0.25
        decision = BeliefPUCTSearch(
            backend=backend,
            evaluator=evaluator,
            config=self.config,
        ).search(obs, self.ledger, temperature=temperature)
        self.last_decision = decision
        if decision.action is None:
            self._last_source = "mcts_fallback"
            return self.policy(obs_dict)
        action = list(decision.action)
        if not is_legal_action(select, action):
            raise ValueError(f"MCTS returned illegal action: {action}")
        self.searchable_decisions += 1
        self._last_source = "mcts"
        return action
