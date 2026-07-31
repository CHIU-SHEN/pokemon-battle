"""Lifecycle-safe adapter around the official branching Search API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SearchStateRef:
    state_id: int
    observation: Any


@dataclass
class SearchBackendStats:
    particles_requested: int = 0
    particles_valid: int = 0
    particles_invalid: int = 0
    begin_errors: int = 0
    step_calls: int = 0
    step_errors: int = 0
    releases: int = 0
    open_states: int = 0


class SearchBackend:
    """Own every native search state created during one root search."""

    def __init__(
        self,
        *,
        sampler: Any,
        search_begin_fn: Callable[..., Any] | None = None,
        search_step_fn: Callable[..., Any] | None = None,
        search_release_fn: Callable[[int], None] | None = None,
        search_end_fn: Callable[[], None] | None = None,
    ) -> None:
        if any(
            item is None
            for item in (search_begin_fn, search_step_fn, search_release_fn, search_end_fn)
        ):
            from cg.api import search_begin, search_end, search_release, search_step

            search_begin_fn = search_begin_fn or search_begin
            search_step_fn = search_step_fn or search_step
            search_release_fn = search_release_fn or search_release
            search_end_fn = search_end_fn or search_end
        self.sampler = sampler
        self._begin = search_begin_fn
        self._step = search_step_fn
        self._release = search_release_fn
        self._end = search_end_fn
        self._open: set[int] = set()
        self._closed = False
        self.stats = SearchBackendStats()

    def __enter__(self) -> "SearchBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def begin_particles(
        self,
        obs: Any,
        ledger: Any,
        *,
        count: int,
    ) -> list[SearchStateRef]:
        if count <= 0:
            raise ValueError("particle count must be positive")
        if self._closed:
            raise RuntimeError("search backend is closed")
        roots = []
        for _ in range(count):
            self.stats.particles_requested += 1
            particle = self.sampler.sample(obs, ledger)
            if not self.sampler.validate(obs, particle):
                self.stats.particles_invalid += 1
                continue
            try:
                state = self._begin(
                    obs,
                    particle.your_deck,
                    particle.your_prize,
                    particle.opponent_deck,
                    particle.opponent_prize,
                    particle.opponent_hand,
                    particle.opponent_active,
                    manual_coin=False,
                )
            except Exception:
                self.stats.begin_errors += 1
                continue
            state_id = int(state.searchId)
            self._open.add(state_id)
            self.stats.particles_valid += 1
            roots.append(SearchStateRef(state_id, state.observation))
        self.stats.open_states = len(self._open)
        return roots

    def step(self, state_id: int, action: tuple[int, ...]) -> SearchStateRef:
        if state_id not in self._open:
            raise ValueError(f"search state is not owned: {state_id}")
        self.stats.step_calls += 1
        try:
            state = self._step(state_id, list(action))
        except Exception:
            self.stats.step_errors += 1
            raise
        child_id = int(state.searchId)
        self._open.add(child_id)
        self.stats.open_states = len(self._open)
        return SearchStateRef(child_id, state.observation)

    def release(self, state_id: int) -> None:
        if state_id not in self._open:
            return
        self._release(state_id)
        self._open.remove(state_id)
        self.stats.releases += 1
        self.stats.open_states = len(self._open)

    def close(self) -> None:
        if self._closed:
            return
        for state_id in sorted(self._open, reverse=True):
            try:
                self._release(state_id)
                self.stats.releases += 1
            finally:
                self._open.discard(state_id)
        self.stats.open_states = 0
        self._end()
        self._closed = True

    def report(self) -> dict[str, int]:
        self.stats.open_states = len(self._open)
        return asdict(self.stats)
