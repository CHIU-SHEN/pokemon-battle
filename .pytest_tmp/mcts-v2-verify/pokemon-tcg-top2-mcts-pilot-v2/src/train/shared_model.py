"""Shared state/option/deck policy-value network for dynamic legal actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SharedModelConfig:
    global_dim: int
    option_dim: int
    max_card_id: int = 1267
    hidden_dim: int = 192
    option_hidden_dim: int = 192
    deck_embedding_dim: int = 64
    dropout: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return total / denom


class SharedPolicyValueNet(nn.Module):
    def __init__(self, config: SharedModelConfig) -> None:
        super().__init__()
        self.config = config
        self.card_embedding = nn.Embedding(config.max_card_id + 1, config.deck_embedding_dim, padding_idx=0)
        self.deck_encoder = _mlp(config.deck_embedding_dim * 2, config.hidden_dim, config.hidden_dim, config.dropout)
        self.state_encoder = _mlp(config.global_dim, config.hidden_dim, config.hidden_dim, config.dropout)
        self.option_encoder = _mlp(config.option_dim, config.option_hidden_dim, config.hidden_dim, config.dropout)
        self.context = _mlp(config.hidden_dim * 2, config.hidden_dim, config.hidden_dim, config.dropout)
        self.option_bias = nn.Linear(config.hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh(),
        )

    def encode_decks(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        player_ids = batch["player_deck"].clamp(0, self.config.max_card_id)
        opponent_ids = batch["opponent_deck"].clamp(0, self.config.max_card_id)
        player = _masked_mean(self.card_embedding(player_ids), batch["player_deck_mask"])
        opponent = _masked_mean(self.card_embedding(opponent_ids), batch["opponent_deck_mask"])
        return self.deck_encoder(torch.cat([player, opponent], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        legal_mask = batch["legal_mask"]
        if not legal_mask.any(dim=1).all():
            raise ValueError("each sample requires at least one legal option")
        state = self.state_encoder(batch["global_features"])
        deck = self.encode_decks(batch)
        context = self.context(torch.cat([state, deck], dim=-1))
        options = self.option_encoder(batch["option_features"])
        logits = (options * context.unsqueeze(1)).sum(dim=-1) / (self.config.hidden_dim ** 0.5)
        logits = logits + self.option_bias(options).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        pooled_options = _masked_mean(options, legal_mask)
        value = self.value_head(torch.cat([context, pooled_options], dim=-1)).squeeze(-1)
        return {"policy_logits": logits, "value": value}


def weighted_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    logits = outputs["policy_logits"].float()
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample_policy = -(batch["soft_policy"].float() * log_probs).sum(dim=-1)
    policy_denom = batch["policy_weight"].sum().clamp_min(1e-8)
    policy_loss = (per_sample_policy * batch["policy_weight"]).sum() / policy_denom

    per_sample_value = F.mse_loss(outputs["value"].float(), batch["value_target"].float(), reduction="none")
    value_denom = batch["value_weight"].sum().clamp_min(1e-8)
    value_loss = (per_sample_value * batch["value_weight"]).sum() / value_denom
    total = policy_loss + value_loss
    return {"loss": total, "policy_loss": policy_loss, "value_loss": value_loss}


@torch.no_grad()
def batch_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    target_mask = batch["soft_policy"] > 0
    predicted = outputs["policy_logits"].argmax(dim=-1)
    policy_rows = batch["policy_weight"] > 0
    hits = target_mask.gather(1, predicted.unsqueeze(1)).squeeze(1) & policy_rows
    value_rows = batch["value_weight"] > 0
    value_squared_error = (outputs["value"] - batch["value_target"]).square()
    return {
        "policy_correct": float(hits.sum().item()),
        "policy_count": float(policy_rows.sum().item()),
        "value_squared_error": float(value_squared_error[value_rows].sum().item()),
        "value_count": float(value_rows.sum().item()),
    }
