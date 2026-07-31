from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeSearchState:
    searchId: int
    observation: object


class FakeSampler:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls = 0

    def sample(self, obs: object, ledger: object) -> object:
        self.calls += 1
        return type(
            "Particle",
            (),
            {
                "your_deck": [1],
                "your_prize": [2],
                "opponent_deck": [3],
                "opponent_prize": [4],
                "opponent_hand": [5],
                "opponent_active": [6],
            },
        )()

    def validate(self, obs: object, particle: object) -> bool:
        return self.valid


def test_backend_releases_every_state_exactly_once_and_close_is_idempotent() -> None:
    from src.rl.search_backend import SearchBackend

    released: list[int] = []
    ended: list[bool] = []
    next_id = iter((10, 11, 12))

    def begin(*args: object, **kwargs: object) -> FakeSearchState:
        return FakeSearchState(next(next_id), {"current": {"yourIndex": 0}})

    def step(state_id: int, action: list[int]) -> FakeSearchState:
        assert state_id == 10
        assert action == [1]
        return FakeSearchState(next(next_id), {"current": {"yourIndex": 1}})

    backend = SearchBackend(
        sampler=FakeSampler(),
        search_begin_fn=begin,
        search_step_fn=step,
        search_release_fn=released.append,
        search_end_fn=lambda: ended.append(True),
    )

    roots = backend.begin_particles(object(), None, count=1)
    child = backend.step(roots[0].state_id, (1,))
    backend.release(child.state_id)
    backend.release(child.state_id)
    backend.close()
    backend.close()

    assert released == [11, 10]
    assert ended == [True]
    assert backend.stats.open_states == 0


def test_backend_counts_invalid_particles_without_serializing_them() -> None:
    from src.rl.search_backend import SearchBackend

    backend = SearchBackend(
        sampler=FakeSampler(valid=False),
        search_begin_fn=lambda *args, **kwargs: None,
        search_step_fn=lambda *args, **kwargs: None,
        search_release_fn=lambda state_id: None,
        search_end_fn=lambda: None,
    )

    assert backend.begin_particles(object(), None, count=3) == []
    report = backend.report()

    assert report["particles_requested"] == 3
    assert report["particles_invalid"] == 3
    assert not {"your_deck", "opponent_deck", "opponent_hand"} & set(report)


def test_context_manager_closes_states_after_exception() -> None:
    from src.rl.search_backend import SearchBackend

    released: list[int] = []
    ended: list[bool] = []
    backend = SearchBackend(
        sampler=FakeSampler(),
        search_begin_fn=lambda *args, **kwargs: FakeSearchState(7, object()),
        search_step_fn=lambda *args, **kwargs: FakeSearchState(8, object()),
        search_release_fn=released.append,
        search_end_fn=lambda: ended.append(True),
    )

    try:
        with backend:
            backend.begin_particles(object(), None, count=1)
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert released == [7]
    assert ended == [True]
