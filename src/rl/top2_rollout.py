"""Branch-safe rollout primitives for Top2 reinforcement learning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from src.rl.top2_ppo import generalized_advantage_estimate


def stable_game_split(game_id: str) -> str:
    """Assign a whole game to stable 80/10/10 train/valid/test splits."""

    bucket = int(hashlib.sha256(game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "valid" if bucket < 90 else "test"


def finalize_trajectory(
    decisions: list[dict],
    *,
    game_id: str,
    deck_id: str,
    result: int,
    learner_side: int,
    gamma: float,
    gae_lambda: float,
) -> list[dict]:
    """Attach one terminal outcome and branch identity to sampled decisions."""

    if not decisions:
        return []
    outcome = 0.0 if result == 2 else 1.0 if result == learner_side else -1.0
    rewards = [0.0] * len(decisions)
    rewards[-1] = outcome
    dones = [False] * len(decisions)
    dones[-1] = True
    values = [float(row["value"]) for row in decisions]
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    split = stable_game_split(game_id)
    finalized = []
    for index, source in enumerate(decisions):
        row = dict(source)
        row.update(
            {
                "game_id": game_id,
                "deck_id": deck_id,
                "split": split,
                "reward": rewards[index],
                "done": dones[index],
                "advantage": advantages[index],
                "return": returns[index],
            }
        )
        finalized.append(row)
    return finalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Top2RolloutAgent:
    """Sample legal Adapter actions and retain the behavior-policy statistics."""

    def __init__(
        self,
        candidate_id: str,
        deck_id: str,
        *,
        project_root: Path,
        device: str = "cpu",
        seed: int = 20260729,
        temperature: float = 1.0,
        ppo_checkpoint: Path | None = None,
        deterministic: bool = False,
        record_decisions: bool = True,
    ) -> None:
        from src.arena.adapter_agent import AdapterArenaAgent

        self.candidate_id = candidate_id
        self.deck_id = deck_id
        self.project_root = Path(project_root)
        self.device = torch.device(device)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.deterministic = deterministic
        self.record_decisions = record_decisions
        self.base_agent = AdapterArenaAgent(candidate_id, project_root=self.project_root, device=self.device)
        self.model = self.base_agent.model
        self.deck = list(self.base_agent.deck)
        self.deck_path = self.base_agent.deck_path
        self.adapter_path = self.base_agent.adapter_path
        self.generator = torch.Generator(device=self.device.type)
        self.generator.manual_seed(seed)
        self.decisions: list[dict[str, Any]] = []
        self._last_source = "initialized"
        if ppo_checkpoint is not None:
            self._load_ppo_checkpoint(Path(ppo_checkpoint))

    def _load_ppo_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "top2_ppo_checkpoint_v1":
            raise ValueError("unsupported PPO checkpoint schema")
        if checkpoint.get("candidate_id") != self.candidate_id:
            raise ValueError("PPO checkpoint candidate mismatch")
        if checkpoint.get("deck_id") != self.deck_id:
            raise ValueError("PPO checkpoint deck_id mismatch")
        state = checkpoint["adapter_state"]
        self.model.adapter.load_state_dict(state["adapter"], strict=True)
        self.model.policy_delta.load_state_dict(state["policy_delta"], strict=True)
        self.model.value_delta.load_state_dict(state["value_delta"], strict=True)
        self.model.to(self.device).eval()

    def reset_trajectory(self) -> None:
        self.decisions.clear()

    def action_source(self) -> str:
        return self._last_source

    def evaluate_policy_value(self, obs_or_dict: Any) -> tuple[Any, list[float], float, dict[str, Any]]:
        """Evaluate one public observation without sampling an action."""

        from agent.parser import parse_observation
        from src.train.features import sample_features

        parsed = parse_observation(obs_or_dict)
        global_vec, option_vecs, _, _ = sample_features(parsed, self.base_agent.tags)
        if not option_vecs:
            return parsed, [], 0.0, {
                "global_features": global_vec,
                "option_features": option_vecs,
                "legal_mask": [],
                "player_deck": list(self.deck),
            }
        option_count = len(option_vecs)
        batch = {
            "global_features": torch.tensor([global_vec], dtype=torch.float32, device=self.device),
            "option_features": torch.tensor([option_vecs], dtype=torch.float32, device=self.device),
            "legal_mask": torch.ones((1, option_count), dtype=torch.bool, device=self.device),
            "player_deck": torch.tensor([self.deck], dtype=torch.long, device=self.device),
            "player_deck_mask": torch.ones((1, len(self.deck)), dtype=torch.bool, device=self.device),
            "opponent_deck": torch.zeros((1, 1), dtype=torch.long, device=self.device),
            "opponent_deck_mask": torch.zeros((1, 1), dtype=torch.bool, device=self.device),
        }
        with torch.inference_mode():
            output = self.model(batch)
        return parsed, output["policy_logits"][0].detach().cpu().tolist(), float(
            output["value"][0].item()
        ), {
            "global_features": global_vec,
            "option_features": option_vecs,
            "legal_mask": [True] * option_count,
            "player_deck": list(self.deck),
        }

    def __call__(self, obs_dict: dict | None) -> list[int]:
        if obs_dict is None or obs_dict.get("select") is None:
            self._last_source = "deck"
            return list(self.deck)

        from agent.fallback import is_legal_action
        from agent.parser import parse_observation
        from cg.api import to_observation_class
        from src.train.features import sample_features

        obs = to_observation_class(obs_dict)
        select = obs.select
        if select is not None and select.minCount == select.maxCount == 1 and len(select.option) > 1:
            try:
                parsed = parse_observation(obs_dict)
                global_vec, option_vecs, _, _ = sample_features(parsed, self.base_agent.tags)
                option_count = len(option_vecs)
                batch = {
                    "global_features": torch.tensor([global_vec], dtype=torch.float32, device=self.device),
                    "option_features": torch.tensor([option_vecs], dtype=torch.float32, device=self.device),
                    "legal_mask": torch.ones((1, option_count), dtype=torch.bool, device=self.device),
                    "player_deck": torch.tensor([self.deck], dtype=torch.long, device=self.device),
                    "player_deck_mask": torch.ones((1, len(self.deck)), dtype=torch.bool, device=self.device),
                    "opponent_deck": torch.zeros((1, 1), dtype=torch.long, device=self.device),
                    "opponent_deck_mask": torch.zeros((1, 1), dtype=torch.bool, device=self.device),
                }
                with torch.inference_mode():
                    output = self.model(batch)
                    logits = output["policy_logits"][0] / self.temperature
                    probs = torch.softmax(logits, dim=-1)
                    index = int(
                        probs.argmax().item()
                        if self.deterministic
                        else torch.multinomial(probs, 1, generator=self.generator).item()
                    )
                    log_prob = float(torch.log(probs[index].clamp_min(1e-12)).item())
                    value = float(output["value"][0].item())
                    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
                action = [index]
                if not is_legal_action(obs.select, action):
                    raise ValueError(f"sampled illegal action: {action}")
                if self.record_decisions:
                    self.decisions.append(
                        {
                            "step": len(self.decisions),
                            "global_features": global_vec,
                            "option_features": option_vecs,
                            "legal_mask": [True] * option_count,
                            "player_deck": self.deck,
                            "opponent_deck": [],
                            "action": index,
                            "old_log_prob": log_prob,
                            "value": value,
                            "entropy": entropy,
                            "confidence": float(probs.max().item()),
                            "observation": obs_dict,
                        }
                    )
                self._last_source = "ppo_policy"
                return action
            except Exception:
                self._last_source = "ppo_fallback"
        action = self.base_agent(obs_dict)
        self._last_source = self.base_agent.action_source()
        return action
