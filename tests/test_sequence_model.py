"""SL-1 temporal inputs, warm start, masks, and backward smoke."""

from __future__ import annotations

import torch

from src.train.sequence_data import collate_sequence_windows
from src.train.sequence_model import (
    SequenceModelConfig,
    SequencePolicyValueNet,
    endpoint_targets,
    initialize_from_sl0,
)
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, weighted_losses


def row(sample: str, step: int, turn: int, action: int) -> dict:
    return {
        "sample_id": sample, "game_id": "g", "step": step, "turn": turn,
        "features": [float(step) / 10, float(step) / 10] + [0.0] * 25,
        "select": {"option_count": 2},
        "option_features": [[0.0, 1.0], [1.0, 0.0]],
        "legal_mask": [True, True], "observed_action": [action],
        "public_history": [{"type": 1}] if step else [],
        "deck": {"player": {"cards": [1]}, "opponent": {"cards": [2]}},
        "value_target": 1.0,
        "supervision": {"soft_policy": [1.0, 0.0], "policy_source": "test",
                        "head_weights": {"policy": 1.0, "value": 0.5}},
        "quality": {"forced_single_option": False},
    }


def main() -> int:
    rows = [row("a", 0, 0, 1), row("b", 1, 0, 0), row("c", 2, 1, 1)]
    window = {"trajectory_id": "t", "game_id": "g", "player": 0, "split": "train",
              "window_length": 3, "valid_length": 3, "start_position": 0, "end_position": 2,
              "turns": [0, 0, 1], "rows": rows, "previous_row": None}
    batch = collate_sequence_windows([window])
    shared = SharedPolicyValueNet(SharedModelConfig(
        global_dim=27, option_dim=2, hidden_dim=16, option_hidden_dim=16,
        deck_embedding_dim=8, dropout=0.0,
    ))
    model = SequencePolicyValueNet(SequenceModelConfig(
        global_dim=27, option_dim=2, transition_dim=24, hidden_dim=16,
        gru_hidden_dim=16, option_hidden_dim=16, deck_embedding_dim=8, dropout=0.0,
    ))
    initialize_from_sl0(model, {"model_state": shared.state_dict()})
    outputs = model(batch)
    assert outputs["policy_logits"].shape == (1, 2)
    assert outputs["value"].shape == (1,)
    loss = weighted_losses(outputs, endpoint_targets(batch))["loss"]
    loss.backward()
    assert model.temporal_projection.weight.grad is not None
    print("OK: SL-1 transition/action GRU forward and backward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
