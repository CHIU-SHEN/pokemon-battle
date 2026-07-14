"""Bounded belief-guided root search using the official Search API."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any

from cg.api import Observation, search_begin, search_end, search_release, search_step, to_observation_class

from .action_gen import ActionCandidate, ActionGenerator
from .belief import BeliefSampler
from .fallback import is_legal_action
from .parser import GameLedger, ParsedState
from .rules import choose_action
from .value import evaluate_observation


@dataclass
class SearchConfig:
    enabled: bool = True
    max_candidates: int = 6
    particles: int = 3
    node_budget: int = 64
    time_budget_sec: float = 0.035
    gumbel_scale: float = 0.15
    opponent_rho: float = 0.35
    switch_margin: float = 175.0


@dataclass
class SearchStats:
    calls: int = 0
    used_search: int = 0
    fallbacks: int = 0
    errors: int = 0
    nodes: int = 0
    last_error: str | None = None
    last_report: dict[str, Any] = field(default_factory=dict)


class SearchManager:
    def __init__(self, deck: list[int] | None = None, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.sampler = BeliefSampler(deck=deck)
        self.generator = ActionGenerator()
        self.stats = SearchStats()
        self.rng = random.Random(20260706)

    def should_search(self, parsed: ParsedState, obs: Observation) -> bool:
        if not self.config.enabled or parsed.select is None:
            return False
        if not getattr(obs, "search_begin_input", None):
            return False
        if len(parsed.select.options) <= 1:
            return False
        if parsed.select.min_count != parsed.select.max_count and parsed.select.max_count > 2:
            return False
        return parsed.select.context in {0, 3, 4, 7, 21, 22, 35, 37, 41}

    def _score_candidate_once(
        self,
        obs: Observation,
        root_player: int,
        candidate: ActionCandidate,
        ledger: GameLedger | None,
    ) -> float | None:
        particle = self.sampler.sample(obs, ledger)
        if not self.sampler.validate(obs, particle):
            return None
        root_state_id: int | None = None
        child_state_id: int | None = None
        try:
            root = search_begin(
                obs,
                particle.your_deck,
                particle.your_prize,
                particle.opponent_deck,
                particle.opponent_prize,
                particle.opponent_hand,
                particle.opponent_active,
                manual_coin=False,
            )
            root_state_id = root.searchId
            child = search_step(root.searchId, candidate.as_list())
            child_state_id = child.searchId
            self.stats.nodes += 2
            return evaluate_observation(child.observation, root_player)
        finally:
            if child_state_id is not None:
                try:
                    search_release(child_state_id)
                except Exception:
                    pass
            if root_state_id is not None:
                try:
                    search_release(root_state_id)
                except Exception:
                    pass

    def choose(self, obs_dict: dict, parsed: ParsedState, ledger: GameLedger | None = None) -> list[int]:
        self.stats.calls += 1
        obs = to_observation_class(obs_dict)
        v0_action = choose_action(parsed)
        if parsed.select is None or not is_legal_action(parsed.select, v0_action):
            return v0_action
        if not self.should_search(parsed, obs):
            return v0_action

        started = time.perf_counter()
        candidates = self.generator.generate(parsed, self.config.max_candidates)
        candidates = [c for c in candidates if is_legal_action(parsed.select, c.as_list())]
        if len(candidates) <= 1:
            return v0_action

        root_player = parsed.current_player
        values: dict[tuple[int, ...], list[float]] = {c.action: [] for c in candidates}
        priors = {c.action: c.prior for c in candidates}
        errors = 0
        local_nodes = 0

        try:
            for _ in range(self.config.particles):
                for candidate in candidates:
                    if local_nodes >= self.config.node_budget:
                        break
                    if time.perf_counter() - started >= self.config.time_budget_sec:
                        break
                    try:
                        score = self._score_candidate_once(obs, root_player, candidate, ledger)
                    except Exception as exc:
                        errors += 1
                        self.stats.last_error = f"{type(exc).__name__}: {exc}"
                        score = None
                    if score is not None:
                        local_nodes += 2
                        u = max(1e-9, min(1.0 - 1e-9, self.rng.random()))
                        gumbel = -math.log(-math.log(u)) * self.config.gumbel_scale
                        values[candidate.action].append(score + gumbel)
                if time.perf_counter() - started >= self.config.time_budget_sec:
                    break
        finally:
            try:
                search_end()
            except Exception:
                pass

        scored = []
        for candidate in candidates:
            vals = values[candidate.action]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            worst = min(vals)
            risk_score = (1.0 - self.config.opponent_rho) * mean + self.config.opponent_rho * worst
            risk_score += priors[candidate.action] * 25.0
            scored.append((risk_score, candidate))

        self.stats.errors += errors
        self.stats.nodes += local_nodes
        if not scored:
            self.stats.fallbacks += 1
            return v0_action

        scored.sort(key=lambda x: x[0], reverse=True)
        def score_report(score: float, candidate: ActionCandidate) -> dict[str, Any]:
            samples = values[candidate.action]
            return {
                "action": candidate.as_list(),
                "source": candidate.source,
                "score": score,
                "visits": len(samples),
                "mean_score": sum(samples) / len(samples) if samples else None,
                "worst_score": min(samples) if samples else None,
            }

        v0_key = tuple(v0_action)
        v0_score = next((score for score, candidate in scored if candidate.action == v0_key), None)
        if scored[0][1].action != v0_key and (v0_score is None or scored[0][0] < v0_score + self.config.switch_margin):
            self.stats.fallbacks += 1
            self.stats.last_report = {
                "elapsed_sec": time.perf_counter() - started,
                "candidate_count": len(candidates),
                "scored_count": len(scored),
                "errors": errors,
                "chosen_source": "v0_margin_guard",
                "v0_action": v0_action,
                "chosen_action": v0_action,
                "scores": [score_report(score, c) for score, c in scored[:5]],
            }
            return v0_action

        best = scored[0][1].as_list()
        if not is_legal_action(parsed.select, best):
            self.stats.fallbacks += 1
            return v0_action

        self.stats.used_search += 1
        self.stats.last_report = {
            "elapsed_sec": time.perf_counter() - started,
            "candidate_count": len(candidates),
            "scored_count": len(scored),
            "errors": errors,
            "chosen_source": scored[0][1].source,
            "v0_action": v0_action,
            "chosen_action": best,
            "scores": [score_report(score, c) for score, c in scored[:5]],
        }
        return best
