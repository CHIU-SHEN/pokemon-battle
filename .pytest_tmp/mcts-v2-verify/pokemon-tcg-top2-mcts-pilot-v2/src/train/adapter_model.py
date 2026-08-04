"""Frozen SL-0 trunk with a small trainable deck Adapter."""

from __future__ import annotations

import torch
from torch import nn

from src.train.shared_model import SharedPolicyValueNet, _masked_mean


class DeckAdapterPolicyValueNet(nn.Module):
    def __init__(self, base: SharedPolicyValueNet, bottleneck_dim: int = 32) -> None:
        super().__init__()
        self.base = base
        hidden = base.config.hidden_dim
        self.adapter = nn.Sequential(
            nn.Linear(hidden, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden, bias=False),
        )
        self.policy_delta = nn.Linear(hidden, 1, bias=False)
        self.value_delta = nn.Sequential(nn.Linear(hidden * 2, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.policy_delta.weight)
        nn.init.zeros_(self.value_delta[-1].weight)
        nn.init.zeros_(self.value_delta[-1].bias)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        legal_mask = batch["legal_mask"]
        with torch.no_grad():
            state = self.base.state_encoder(batch["global_features"])
            deck = self.base.encode_decks(batch)
            context = self.base.context(torch.cat([state, deck], dim=-1))
            options = self.base.option_encoder(batch["option_features"])
        adapted = context + self.adapter(context)
        logits = (options * adapted.unsqueeze(1)).sum(dim=-1) / (self.base.config.hidden_dim ** 0.5)
        logits = logits + self.base.option_bias(options).squeeze(-1) + self.policy_delta(options).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        pooled = _masked_mean(options, legal_mask)
        with torch.no_grad():
            base_value = self.base.value_head(torch.cat([context, pooled], dim=-1)).squeeze(-1)
        value = (base_value + self.value_delta(torch.cat([adapted, pooled], dim=-1)).squeeze(-1)).clamp(-1.0, 1.0)
        return {"policy_logits": logits, "value": value}

    def adapter_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "adapter": self.adapter.state_dict(),
            "policy_delta": self.policy_delta.state_dict(),
            "value_delta": self.value_delta.state_dict(),
        }
