"""Leakage and milestone checks for Combo weak labels."""

from __future__ import annotations

from collections import Counter, deque

from scripts.build_combo_labels import combo_flags, future_flags, history_vector, selected_milestones


def sample(option_type: int, tags: list[str]) -> dict:
    return {
        "options": [{"option_type": option_type, "tags": tags}],
        "observed_action": [0],
    }


def main() -> int:
    rows = [
        sample(7, ["search_pokemon"]),
        sample(9, ["evolution_pokemon"]),
        sample(8, ["basic_energy", "attach_energy"]),
        sample(13, ["main_attacker"]),
    ]
    events = [selected_milestones(row) for row in rows]
    assert events[0]["play"] and events[0]["search"]
    future = future_flags(events, 0, 3)
    combos = combo_flags(events[0], future)
    assert combos["setup_then_evolve"] and combos["search_then_play_or_evolve"]
    assert not combos["energy_then_attack"]
    empty = history_vector(deque(maxlen=8), Counter(), {"attach": 32, "evolve": 32, "attack": 32})
    assert all(value == 0.0 for value in empty[:-3])
    # Future changes labels but cannot affect history input features.
    altered = events[:1] + [selected_milestones(sample(14, [])) for _ in range(3)]
    assert future_flags(events, 0, 3) != future_flags(altered, 0, 3)
    assert empty == history_vector(deque(maxlen=8), Counter(), {"attach": 32, "evolve": 32, "attack": 32})
    print("OK: Combo milestones, future targets and history leakage boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
