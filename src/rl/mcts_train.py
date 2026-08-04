"""AlphaZero-style losses and loaders for Top2 MCTS targets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def _joint_logits(option_log_probs: torch.Tensor, actions: list[tuple[int, ...]]) -> torch.Tensor:
    values = []
    for action in actions:
        if not action:
            values.append(option_log_probs.new_tensor(math.log(0.05)))
        else:
            indices = torch.tensor(action, dtype=torch.long, device=option_log_probs.device)
            values.append(option_log_probs.index_select(0, indices).sum() / math.sqrt(len(action)))
    return torch.stack(values)


def mcts_loss(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    reference_logits: torch.Tensor,
    actions: list[list[tuple[int, ...]]],
    policy_targets: list[list[float]],
    value_targets: torch.Tensor,
    legal_mask: torch.Tensor,
    value_coef: float,
    kl_coef: float,
    entropy_coef: float,
) -> dict[str, torch.Tensor]:
    floor = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~legal_mask, floor)
    masked_reference = reference_logits.masked_fill(~legal_mask, floor)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    reference_log_probs = F.log_softmax(masked_reference, dim=-1)
    probs = log_probs.exp()
    policy_losses = []
    for index, sample_actions in enumerate(actions):
        candidate_logits = _joint_logits(log_probs[index], sample_actions)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        target = torch.tensor(
            policy_targets[index],
            dtype=candidate_log_probs.dtype,
            device=candidate_log_probs.device,
        )
        policy_losses.append(-(target * candidate_log_probs).sum())
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = F.mse_loss(values, value_targets)
    reference_kl = (
        probs * (log_probs - reference_log_probs)
    ).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    loss = policy_loss + value_coef * value_loss + kl_coef * reference_kl - entropy_coef * entropy
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "reference_kl": reference_kl,
        "entropy": entropy,
    }


def load_mcts_rows(
    path: Path,
    *,
    branch: str,
    deck_id: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for source in sorted(Path(path).rglob("game_*.json")):
        document = json.loads(source.read_text(encoding="utf-8"))
        if document.get("schema_version") != "top2_mcts_game_v1":
            continue
        for row in document.get("samples") or []:
            if row.get("branch") != branch:
                raise ValueError(f"MCTS sample branch mismatch in {source}")
            if row.get("deck_id") != deck_id:
                raise ValueError(f"MCTS sample deck mismatch in {source}")
            if row.get("split") == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"no MCTS {split} samples found under {path}")
    return rows


def collate_mcts_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_options = max(len(row["option_features"]) for row in rows)
    option_dim = len(rows[0]["option_features"][0])
    max_cards = max(len(row["player_deck"]) for row in rows)
    batch_size = len(rows)
    option_features = torch.zeros((batch_size, max_options, option_dim), dtype=torch.float32)
    legal_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
    player_deck = torch.zeros((batch_size, max_cards), dtype=torch.long)
    player_deck_mask = torch.zeros((batch_size, max_cards), dtype=torch.bool)
    for index, row in enumerate(rows):
        option_count = len(row["option_features"])
        option_features[index, :option_count] = torch.tensor(row["option_features"])
        legal_mask[index, :option_count] = torch.tensor(row["legal_mask"], dtype=torch.bool)
        cards = row["player_deck"]
        player_deck[index, : len(cards)] = torch.tensor(cards, dtype=torch.long)
        player_deck_mask[index, : len(cards)] = True
    return {
        "global_features": torch.tensor([row["global_features"] for row in rows], dtype=torch.float32),
        "option_features": option_features,
        "legal_mask": legal_mask,
        "player_deck": player_deck,
        "player_deck_mask": player_deck_mask,
        "opponent_deck": torch.zeros((batch_size, 1), dtype=torch.long),
        "opponent_deck_mask": torch.zeros((batch_size, 1), dtype=torch.bool),
        "actions": [[tuple(action) for action in row["actions"]] for row in rows],
        "policy_targets": [row["policy_target"] for row in rows],
        "value_targets": torch.tensor([row["value_target"] for row in rows], dtype=torch.float32),
    }


def evaluate_mcts_rows(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    value_coef: float,
    kl_coef: float,
    entropy_coef: float,
) -> dict[str, float]:
    """Evaluate frozen MCTS rows without constructing an autograd graph."""
    if not rows:
        raise ValueError("holdout rows must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sums = {key: 0.0 for key in ("loss", "policy_loss", "value_loss", "reference_kl", "entropy")}
    batches = 0
    model.eval()
    reference.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            raw = collate_mcts_rows(rows[offset: offset + batch_size])
            actions = raw.pop("actions")
            policy_targets = raw.pop("policy_targets")
            value_targets = raw.pop("value_targets").to(device)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw.items()
            }
            output = model(batch)
            reference_output = reference(batch)
            losses = mcts_loss(
                logits=output["policy_logits"],
                values=output["value"],
                reference_logits=reference_output["policy_logits"],
                actions=actions,
                policy_targets=policy_targets,
                value_targets=value_targets,
                legal_mask=batch["legal_mask"],
                value_coef=value_coef,
                kl_coef=kl_coef,
                entropy_coef=entropy_coef,
            )
            for key in sums:
                sums[key] += float(losses[key].item())
            batches += 1
    return {key: value / batches for key, value in sums.items()}
