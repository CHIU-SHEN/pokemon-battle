"""Masked-PPO primitives for branch-bound Top2 policies."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def generalized_advantage_estimate(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute GAE for one ordered trajectory, resetting at terminal states."""

    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal length")
    advantages = [0.0] * len(rewards)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = float(rewards[index]) + gamma * next_value * continuation - float(values[index])
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = float(values[index])
    returns = [advantage + float(value) for advantage, value in zip(advantages, values)]
    return advantages, returns


def validate_training_rows(rows: list[dict], deck_id: str) -> list[dict]:
    """Reject held-out or cross-branch rows before any optimizer sees them."""

    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"PPO training accepts train split only: {row.get('split')!r}")
        if row.get("deck_id") != deck_id:
            raise ValueError(
                f"rollout deck_id mismatch: expected={deck_id!r} actual={row.get('deck_id')!r}"
            )
    return rows


def masked_ppo_loss(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    reference_logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    legal_mask: torch.Tensor,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
) -> dict[str, torch.Tensor]:
    """Compute PPO losses after masking every illegal option."""

    if logits.shape != legal_mask.shape or reference_logits.shape != logits.shape:
        raise ValueError("logits, reference_logits, and legal_mask must have equal shape")
    if not legal_mask.any(dim=-1).all():
        raise ValueError("every sample must expose at least one legal option")
    chosen_is_legal = legal_mask.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    if not chosen_is_legal.all():
        raise ValueError("PPO action is illegal under its saved mask")

    floor = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~legal_mask, floor)
    masked_reference = reference_logits.masked_fill(~legal_mask, floor)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    reference_log_probs = F.log_softmax(masked_reference, dim=-1)
    probs = log_probs.exp()
    chosen_log_probs = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    ratios = (chosen_log_probs - old_log_probs).exp()
    unclipped = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    reference_kl = (
        probs * (log_probs - reference_log_probs)
    ).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy + kl_coef * reference_kl
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": reference_kl,
        "clip_fraction": ((ratios - 1.0).abs() > clip_ratio).float().mean(),
    }


def load_rollout_rows(path: Path, deck_id: str) -> list[dict]:
    """Load branch-bound decisions from one file or a rollout directory."""

    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    rows: list[dict] = []
    for source in files:
        doc = json.loads(source.read_text(encoding="utf-8"))
        if doc.get("schema_version") != "top2_rl_rollout_v1":
            continue
        if doc.get("deck_id") != deck_id:
            raise ValueError(f"rollout file crosses deck stream: {source}")
        rows.extend(doc.get("decisions") or [])
    if not rows:
        raise ValueError(f"no rollout decisions found under {path}")
    return validate_training_rows(rows, deck_id)


def collate_rollout_rows(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable legal-option sets for one PPO minibatch."""

    max_options = max(len(row["option_features"]) for row in rows)
    option_dim = len(rows[0]["option_features"][0])
    max_player_cards = max(len(row["player_deck"]) for row in rows)
    max_opponent_cards = max(1, max(len(row.get("opponent_deck") or []) for row in rows))
    batch_size = len(rows)
    option_features = torch.zeros((batch_size, max_options, option_dim), dtype=torch.float32)
    legal_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
    player_deck = torch.zeros((batch_size, max_player_cards), dtype=torch.long)
    player_deck_mask = torch.zeros((batch_size, max_player_cards), dtype=torch.bool)
    opponent_deck = torch.zeros((batch_size, max_opponent_cards), dtype=torch.long)
    opponent_deck_mask = torch.zeros((batch_size, max_opponent_cards), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row["option_features"])
        option_features[index, :count] = torch.tensor(row["option_features"], dtype=torch.float32)
        legal_mask[index, :count] = torch.tensor(row["legal_mask"], dtype=torch.bool)
        cards = row["player_deck"]
        player_deck[index, : len(cards)] = torch.tensor(cards, dtype=torch.long)
        player_deck_mask[index, : len(cards)] = True
        other = row.get("opponent_deck") or []
        if other:
            opponent_deck[index, : len(other)] = torch.tensor(other, dtype=torch.long)
            opponent_deck_mask[index, : len(other)] = True
    return {
        "global_features": torch.tensor([row["global_features"] for row in rows], dtype=torch.float32),
        "option_features": option_features,
        "legal_mask": legal_mask,
        "player_deck": player_deck,
        "player_deck_mask": player_deck_mask,
        "opponent_deck": opponent_deck,
        "opponent_deck_mask": opponent_deck_mask,
        "actions": torch.tensor([row["action"] for row in rows], dtype=torch.long),
        "old_log_probs": torch.tensor([row["old_log_prob"] for row in rows], dtype=torch.float32),
        "advantages": torch.tensor([row["advantage"] for row in rows], dtype=torch.float32),
        "returns": torch.tensor([row["return"] for row in rows], dtype=torch.float32),
    }
