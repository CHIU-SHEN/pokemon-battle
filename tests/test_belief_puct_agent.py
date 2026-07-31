from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeRef:
    state_id: int
    observation: dict


class FakeBackend:
    def __init__(self, roots: list[FakeRef], transitions: dict[tuple[int, tuple[int, ...]], FakeRef]) -> None:
        self.roots = roots
        self.transitions = transitions
        self.step_calls = 0
        self.closed = False

    def begin_particles(self, obs: object, ledger: object, *, count: int) -> list[FakeRef]:
        return self.roots[:count]

    def step(self, state_id: int, action: tuple[int, ...]) -> FakeRef:
        self.step_calls += 1
        return self.transitions[(state_id, action)]

    def close(self) -> None:
        self.closed = True

    def report(self) -> dict[str, int]:
        return {"step_calls": self.step_calls}


def test_search_uses_exact_simulation_budget_and_returns_visit_target() -> None:
    from src.rl.belief_puct_agent import BeliefPUCTSearch, NodeEvaluation, SearchConfig

    root = FakeRef(1, {"id": 1})
    transitions = {
        (1, (0,)): FakeRef(2, {"id": 2}),
        (1, (1,)): FakeRef(3, {"id": 3}),
    }
    backend = FakeBackend([root], transitions)

    def evaluate(obs: dict) -> NodeEvaluation:
        if obs["id"] == 1:
            return NodeEvaluation(((0,), (1,)), (2.0, 0.0), 0.0, player=0, result=-1)
        value = 0.8 if obs["id"] == 2 else -0.2
        return NodeEvaluation((), (), value, player=1, result=-1)

    decision = BeliefPUCTSearch(
        backend=backend,
        evaluator=evaluate,
        config=SearchConfig(simulations=8, particles=1, max_depth=3, root_noise=False),
    ).search(object(), None, temperature=1.0)

    assert sum(decision.visit_counts.values()) == 8
    assert sum(decision.policy_target.values()) == 1.0
    assert decision.action in {(0,), (1,)}
    assert decision.fallback_reason is None
    assert backend.closed


def test_opponent_node_maximizes_opponent_value_but_same_player_does_not_flip() -> None:
    from src.rl.belief_puct_agent import backup_value_for_player

    assert backup_value_for_player(0.75, root_player=0, parent_player=0) == 0.75
    assert backup_value_for_player(0.75, root_player=0, parent_player=1) == -0.75
    assert backup_value_for_player(-0.4, root_player=1, parent_player=1) == -0.4


def test_missing_particles_returns_structured_fallback_without_policy_target() -> None:
    from src.rl.belief_puct_agent import BeliefPUCTSearch, NodeEvaluation, SearchConfig

    backend = FakeBackend([], {})
    decision = BeliefPUCTSearch(
        backend=backend,
        evaluator=lambda obs: NodeEvaluation(((0,),), (0.0,), 0.0, 0, -1),
        config=SearchConfig(simulations=4, particles=1, max_depth=2),
    ).search(object(), None, temperature=1.0)

    assert decision.action is None
    assert decision.policy_target == {}
    assert decision.fallback_reason == "no_valid_particles"
    assert backend.closed


def test_depth_cap_stops_expansion() -> None:
    from src.rl.belief_puct_agent import BeliefPUCTSearch, NodeEvaluation, SearchConfig

    transitions = {
        (1, (0,)): FakeRef(2, {"id": 2}),
        (2, (0,)): FakeRef(3, {"id": 3}),
        (3, (0,)): FakeRef(4, {"id": 4}),
    }
    backend = FakeBackend([FakeRef(1, {"id": 1})], transitions)

    decision = BeliefPUCTSearch(
        backend=backend,
        evaluator=lambda obs: NodeEvaluation(((0,),), (0.0,), 0.1, player=0, result=-1),
        config=SearchConfig(simulations=3, particles=1, max_depth=1, root_noise=False),
    ).search(object(), None, temperature=0.0)

    assert decision.max_depth_reached <= 1
    assert backend.step_calls == 1
