from __future__ import annotations

import numpy as np
import pytest


def test_puct_prefers_prior_before_any_visit() -> None:
    from src.rl.puct import EdgeStats, select_puct_action

    edges = {
        (0,): EdgeStats(prior=0.2),
        (1,): EdgeStats(prior=0.8),
    }

    assert select_puct_action(edges, c_puct=1.5) == (1,)


def test_puct_uses_backed_up_value() -> None:
    from src.rl.puct import EdgeStats, select_puct_action

    edges = {
        (0,): EdgeStats(prior=0.5, visits=10, value_sum=8.0),
        (1,): EdgeStats(prior=0.5, visits=10, value_sum=-2.0),
    }

    assert select_puct_action(edges, c_puct=0.1) == (0,)


def test_joint_action_priors_are_normalized_and_length_corrected() -> None:
    from src.rl.puct import joint_action_priors

    priors = joint_action_priors(
        option_logits=[2.0, 1.0, 0.0],
        actions=[(0,), (1,), (1, 2)],
    )

    assert set(priors) == {(0,), (1,), (1, 2)}
    assert sum(priors.values()) == pytest.approx(1.0)
    assert priors[(0,)] > priors[(1,)]
    assert all(value > 0.0 for value in priors.values())


def test_visit_distribution_supports_sampling_and_argmax_temperatures() -> None:
    from src.rl.puct import EdgeStats, visit_distribution

    edges = {
        (0,): EdgeStats(prior=0.5, visits=3),
        (1,): EdgeStats(prior=0.5, visits=1),
    }

    soft = visit_distribution(edges, temperature=1.0)
    hard = visit_distribution(edges, temperature=0.0)

    assert soft == pytest.approx({(0,): 0.75, (1,): 0.25})
    assert hard == {(0,): 1.0, (1,): 0.0}


def test_dirichlet_noise_is_seeded_and_normalized() -> None:
    from src.rl.puct import mix_dirichlet_noise

    priors = {(0,): 0.7, (1,): 0.3}
    first = mix_dirichlet_noise(
        priors,
        alpha=0.3,
        epsilon=0.25,
        rng=np.random.default_rng(7),
    )
    second = mix_dirichlet_noise(
        priors,
        alpha=0.3,
        epsilon=0.25,
        rng=np.random.default_rng(7),
    )

    assert first == pytest.approx(second)
    assert sum(first.values()) == pytest.approx(1.0)
    assert first != priors


@pytest.mark.parametrize("temperature", [-1.0, float("nan")])
def test_visit_distribution_rejects_invalid_temperature(temperature: float) -> None:
    from src.rl.puct import EdgeStats, visit_distribution

    with pytest.raises(ValueError, match="temperature"):
        visit_distribution({(0,): EdgeStats(prior=1.0)}, temperature=temperature)
