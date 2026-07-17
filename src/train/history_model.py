"""SL-0-history model and exact warm-start from the frozen SL-0 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, _mlp


@dataclass(frozen=True)
class HistoryModelConfig:
    global_dim: int
    option_dim: int
    history_dim: int = 24
    max_card_id: int = 1267
    hidden_dim: int = 192
    option_hidden_dim: int = 192
    deck_embedding_dim: int = 64
    dropout: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoryPolicyValueNet(SharedPolicyValueNet):
    def __init__(self, config: HistoryModelConfig) -> None:
        shared = SharedModelConfig(
            global_dim=config.global_dim,
            option_dim=config.option_dim,
            max_card_id=config.max_card_id,
            hidden_dim=config.hidden_dim,
            option_hidden_dim=config.option_hidden_dim,
            deck_embedding_dim=config.deck_embedding_dim,
            dropout=config.dropout,
        )
        super().__init__(shared)
        self.history_config = config
        self.config = config
        self.config = config
        self.state_encoder = _mlp(
            config.global_dim + config.history_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history = batch.get("history_features")
        if history is None or history.shape[-1] != self.history_config.history_dim:
            raise ValueError(f"history_features must have dimension {self.history_config.history_dim}")
        original = batch["global_features"]
        augmented = dict(batch)
        augmented["global_features"] = torch.cat([original, history], dim=-1)
        # Call the grandparent implementation with the augmented state vector.
        return SharedPolicyValueNet.forward(self, augmented)


def initialize_from_sl0(model: HistoryPolicyValueNet, checkpoint: dict[str, Any]) -> None:
    """Warm-start all shared weights and zero the new history columns."""
    source = checkpoint["model_state"]
    target = model.state_dict()
    for name, value in source.items():
        if name == "state_encoder.0.weight":
            expected = target[name]
            if value.shape[0] != expected.shape[0] or value.shape[1] != model.history_config.global_dim:
                raise ValueError("SL-0 state encoder is incompatible with the history model")
            expected.zero_()
            expected[:, : value.shape[1]].copy_(value)
        elif name in target:
            if target[name].shape != value.shape:
                raise ValueError(f"incompatible SL-0 tensor: {name}")
            target[name].copy_(value)
    model.load_state_dict(target, strict=True)
