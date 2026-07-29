#!/usr/bin/env python3
"""Online Adapter agent binding, model-action and fallback checks."""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_match import load_agent, validate_action


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "alakazam_neutralization_zone"


def main() -> int:
    agent = load_agent(f"adapter:{CANDIDATE}")
    deck = agent(None)
    assert len(deck) == 60
    expected = [
        int(line)
        for line in (ROOT / "data" / "high_score_decks" / CANDIDATE / "deck.csv")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    assert deck == expected

    fixtures = json.loads((ROOT / "tests" / "fixtures" / "observations.json").read_text(encoding="utf-8"))
    sources: set[str] = set()
    for observation in fixtures[:50]:
        action = agent(observation)
        assert validate_action(observation, action) is None
        sources.add(agent.action_source())
    diagnostics = agent.diagnostics()
    assert diagnostics["candidate_id"] == CANDIDATE
    assert diagnostics["model_calls"] > 0
    assert "adapter" in sources
    assert sources <= {"adapter", "forced", "rules_fallback", "safe_fallback"}
    print(json.dumps({"ok": True, "sources": sorted(sources), **diagnostics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

