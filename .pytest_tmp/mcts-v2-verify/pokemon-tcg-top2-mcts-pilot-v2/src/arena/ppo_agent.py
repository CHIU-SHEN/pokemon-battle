"""Deterministic Arena agent for a branch-bound Top2 PPO checkpoint."""

from __future__ import annotations

from pathlib import Path

from src.rl.top2_rollout import Top2RolloutAgent


class PPOArenaAgent(Top2RolloutAgent):
    def __init__(
        self,
        candidate_id: str,
        deck_id: str,
        checkpoint: Path,
        *,
        project_root: Path,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            candidate_id,
            deck_id,
            project_root=project_root,
            device=device,
            ppo_checkpoint=checkpoint,
            deterministic=True,
            record_decisions=False,
        )
