"""CPU smoke checks for the SL-0-shared data/model path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sample(sample_id: str, split: str, options: int) -> dict:
    option_dim = 5
    return {
        "sample_id": sample_id,
        "game_id": sample_id.split(":")[0],
        "split": split,
        "features": [0.1, 0.2, 0.3],
        "select": {"option_count": options},
        "option_features": [[float(i), 0.0, 1.0, 0.5, -0.5] for i in range(options)],
        "legal_mask": [True] * options,
        "deck": {
            "player": {"cards": [1, 2, 3]},
            "opponent": {"cards": [4, 5]},
        },
        "value_target": 1.0,
        "supervision": {
            "soft_policy": [1.0] + [0.0] * (options - 1),
            "head_weights": {"policy": 1.0, "value": 0.5, "risk": 0.0},
        },
    }


def main() -> int:
    if importlib.util.find_spec("torch") is None:
        print("SKIP: PyTorch is not installed in the local pokemon-tcg environment")
        return 0

    import torch
    from src.train.shared_data import TrainingJsonlDataset, collate_training_rows
    from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, weighted_losses

    rows = [sample("g1:s1", "train", 2), sample("g2:s1", "train", 4)]
    batch = collate_training_rows(rows)
    assert batch["option_features"].shape == (2, 4, 5)
    assert batch["legal_mask"].tolist() == [[True, True, False, False], [True, True, True, True]]

    model = SharedPolicyValueNet(SharedModelConfig(global_dim=3, option_dim=5, hidden_dim=16, option_hidden_dim=16, deck_embedding_dim=8, dropout=0.0))
    outputs = model(batch)
    assert outputs["policy_logits"].shape == (2, 4)
    assert outputs["value"].shape == (2,)
    assert outputs["policy_logits"][0, 2].item() < -1e20
    losses = weighted_losses(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    with tempfile.TemporaryDirectory(prefix="shared_training_") as tmp:
        data = Path(tmp) / "samples.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows + [sample("g3:s1", "valid", 3)]), encoding="utf-8")
        streamed = list(TrainingJsonlDataset(data, "train", shuffle_buffer=2, seed=7))
        assert {row["sample_id"] for row in streamed} == {"g1:s1", "g2:s1"}
        checkpoint = Path(tmp) / "model.pt"
        torch.save({"config": model.config.to_dict(), "state": model.state_dict()}, checkpoint)
        restored_doc = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored = SharedPolicyValueNet(SharedModelConfig(**restored_doc["config"]))
        restored.load_state_dict(restored_doc["state"])
        restored.eval()
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(batch)["policy_logits"], restored(batch)["policy_logits"])
    print("OK: SL-0-shared streaming, mask, loss, backward and reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
