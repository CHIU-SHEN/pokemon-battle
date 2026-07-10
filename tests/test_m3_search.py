#!/usr/bin/env python3
"""M3 belief, action generation, and Search API smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from agent.action_gen import ActionGenerator  # noqa: E402
from agent.belief import BeliefSampler, read_deck  # noqa: E402
from agent.fallback import is_legal_action  # noqa: E402
from agent.parser import GameLedger, parse_observation  # noqa: E402
from agent.search import SearchConfig, SearchManager  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "observations.json"


def main() -> int:
    observations = json.loads(FIXTURES.read_text(encoding="utf-8"))
    sampler = BeliefSampler(deck=read_deck())
    generator = ActionGenerator()
    ledger = GameLedger()
    sampled = 0
    generated = 0
    searched = 0

    for obs_dict in observations:
        obs = to_observation_class(obs_dict)
        parsed = parse_observation(obs)
        ledger.update(parsed)
        particle = sampler.sample(obs, ledger)
        assert sampler.validate(obs, particle), "belief particle failed validation"
        sampled += 1

        candidates = generator.generate(parsed, k=8)
        assert candidates, "candidate generator returned no actions"
        for candidate in candidates:
            assert is_legal_action(parsed.select, candidate.as_list()), f"illegal candidate: {candidate}"
        generated += len(candidates)

    manager = SearchManager(
        deck=read_deck(),
        config=SearchConfig(enabled=True, max_candidates=3, particles=1, node_budget=16, time_budget_sec=0.05),
    )
    for obs_dict in observations:
        parsed = parse_observation(obs_dict)
        action = manager.choose(obs_dict, parsed, ledger)
        assert is_legal_action(parsed.select, action), "search manager returned illegal action"
        if manager.stats.used_search:
            searched += 1
            break

    assert sampled >= 50, "expected 50 sampled fixtures"
    assert generated >= 50, "expected candidates for fixtures"
    print(f"OK: sampled={sampled}, candidates={generated}, search_used={manager.stats.used_search}, searched={searched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

