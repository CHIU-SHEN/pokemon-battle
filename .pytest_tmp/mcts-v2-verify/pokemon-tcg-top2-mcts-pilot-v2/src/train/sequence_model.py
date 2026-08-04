"""SL-1 GRU over current state, visible state deltas, and the prior own action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, _masked_mean, _mlp


@dataclass(frozen=True)
class SequenceModelConfig(SharedModelConfig):
    transition_dim: int = 24
    gru_hidden_dim: int = 192

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequencePolicyValueNet(SharedPolicyValueNet):
    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__(config)
        self.sequence_config = config
        self.transition_encoder = _mlp(
            config.transition_dim, config.hidden_dim, config.hidden_dim, config.dropout
        )
        self.previous_action_encoder = _mlp(
            config.option_dim, config.hidden_dim, config.hidden_dim, config.dropout
        )
        self.gru = nn.GRU(
            input_size=config.hidden_dim * 3,
            hidden_size=config.gru_hidden_dim,
            batch_first=True,
        )
        self.temporal_projection = nn.Linear(config.gru_hidden_dim, config.hidden_dim, bias=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        flat = batch["flat_batch"]
        batch_size, sequence_length = batch["valid_mask"].shape
        state = self.state_encoder(flat["global_features"])
        deck = self.encode_decks(flat)
        base_context = self.context(torch.cat([state, deck], dim=-1))
        transition = self.transition_encoder(batch["transition_features"])
        previous_action = self.previous_action_encoder(batch["previous_action_features"])
        temporal_input = torch.cat([base_context, transition, previous_action], dim=-1)
        padded = temporal_input.new_zeros((batch_size, sequence_length, temporal_input.shape[-1]))
        positions = batch["sequence_positions"]
        for batch_index in range(batch_size):
            selected = temporal_input[positions[:, 0] == batch_index]
            padded[batch_index, : selected.shape[0]] = selected
        packed = nn.utils.rnn.pack_padded_sequence(
            padded,
            batch["valid_mask"].sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        temporal, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=sequence_length
        )
        endpoint_time = batch["valid_mask"].sum(dim=1) - 1
        temporal_endpoint = temporal[torch.arange(batch_size, device=temporal.device), endpoint_time]
        endpoint_indices = batch["endpoint_flat_indices"]
        context = base_context[endpoint_indices] + self.temporal_projection(temporal_endpoint)

        options = self.option_encoder(flat["option_features"][endpoint_indices])
        legal_mask = flat["legal_mask"][endpoint_indices]
        logits = (options * context.unsqueeze(1)).sum(dim=-1) / (self.config.hidden_dim ** 0.5)
        logits = logits + self.option_bias(options).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        pooled_options = _masked_mean(options, legal_mask)
        value = self.value_head(torch.cat([context, pooled_options], dim=-1)).squeeze(-1)
        return {"policy_logits": logits, "value": value}


def endpoint_targets(batch: dict[str, Any]) -> dict[str, Any]:
    """Expose endpoint supervision in the shape expected by shared losses."""
    flat = batch["flat_batch"]
    indices = batch["endpoint_flat_indices"]
    result = dict(flat)
    for key in ("soft_policy", "policy_weight", "value_target", "value_weight", "legal_mask"):
        result[key] = flat[key][indices]
    return result


def initialize_from_sl0(model: SequencePolicyValueNet, checkpoint: dict[str, Any]) -> None:
    """Copy the complete SL-0 network and start with a zero temporal residual."""
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed = {
        name for name in missing
        if name.startswith(("transition_encoder.", "previous_action_encoder.", "gru.", "temporal_projection."))
    }
    if set(missing) != allowed or unexpected:
        raise ValueError(f"incompatible SL-0 checkpoint: missing={missing}, unexpected={unexpected}")
    nn.init.zeros_(model.temporal_projection.weight)
