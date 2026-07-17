"""SL-0-history warm-start, history sensitivity and one-step training smoke."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from src.train.history_model import HistoryModelConfig, HistoryPolicyValueNet, initialize_from_sl0
from src.train.shared_data import collate_training_rows
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, weighted_losses


def row(sample_id: str, history: list[float]) -> dict:
    return {
        "sample_id": sample_id,
        "game_id": "g",
        "features": [0.1, 0.2, 0.3],
        "history_features": history,
        "select": {"option_count": 2},
        "option_features": [[0.0, 1.0], [1.0, 0.0]],
        "legal_mask": [True, True],
        "deck": {"player": {"cards": [1, 2]}, "opponent": {"cards": [3]}},
        "value_target": 1.0,
        "supervision": {"soft_policy": [1.0, 0.0], "policy_source": "test", "head_weights": {"policy": 1.0, "value": 0.5}},
        "quality": {"forced_single_option": False},
    }


def main() -> int:
    torch.manual_seed(7)
    shared = SharedPolicyValueNet(SharedModelConfig(global_dim=3, option_dim=2, hidden_dim=16, option_hidden_dim=16, deck_embedding_dim=8, dropout=0.0))
    checkpoint = {"model_state": shared.state_dict()}
    history = HistoryPolicyValueNet(HistoryModelConfig(global_dim=3, option_dim=2, history_dim=4, hidden_dim=16, option_hidden_dim=16, deck_embedding_dim=8, dropout=0.0))
    initialize_from_sl0(history, checkpoint)
    zero_batch = collate_training_rows([row("a", [0.0] * 4)])
    shared_batch = {key: value for key, value in zero_batch.items() if key != "history_features"}
    shared.eval(); history.eval()
    with torch.no_grad():
        expected = shared(shared_batch)
        actual = history(zero_batch)
    assert torch.allclose(expected["policy_logits"], actual["policy_logits"], atol=1e-6)
    assert torch.allclose(expected["value"], actual["value"], atol=1e-6)
    # History columns begin at zero, then receive gradient on a real step.
    history.train()
    batch = collate_training_rows([row("b", [1.0, 0.5, 0.0, 0.25])])
    loss = weighted_losses(history(batch), batch)["loss"]
    loss.backward()
    gradient = history.state_encoder[0].weight.grad[:, 3:]
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    print("OK: SL-0-history exact warm-start, collate, gradient and backward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
