"""Self-contained inference smoke for the primary Top2 rollout agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval.run_match import validate_action
from src.rl.top2_rollout import Top2RolloutAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    project_root = Path(os.environ.get("PTCG_PROJECT_ROOT", ROOT)).resolve()
    config = json.loads((ROOT / "config/top2_rl_policy.json").read_text(encoding="utf-8"))
    branch = config["branches"][0]
    agent = Top2RolloutAgent(
        branch["candidate_id"],
        branch["deck_id"],
        project_root=project_root,
        device="cpu",
        deterministic=True,
    )
    assert len(agent(None)) == 60
    fixtures = json.loads((ROOT / "tests/fixtures/observations.json").read_text(encoding="utf-8"))
    sources = set()
    for observation in fixtures[:50]:
        action = agent(observation)
        assert validate_action(observation, action) is None
        sources.add(agent.action_source())
    assert agent.decisions
    assert "ppo_policy" in sources
    print(json.dumps({"ok": True, "decisions": len(agent.decisions), "sources": sorted(sources)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
