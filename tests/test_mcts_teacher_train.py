from __future__ import annotations

import pytest
import torch


class _TeacherModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        self.adapter = torch.nn.Linear(1, 1)
        self.policy_delta = torch.nn.Linear(1, 1)
        self.value_delta = torch.nn.Linear(1, 1)


def test_configure_teacher_parameters_freezes_only_base() -> None:
    from scripts.train_top2_mcts import configure_teacher_parameters

    model = _TeacherModel()
    names, parameters = configure_teacher_parameters(model)

    assert names == [
        "adapter.bias",
        "adapter.weight",
        "policy_delta.bias",
        "policy_delta.weight",
        "value_delta.bias",
        "value_delta.weight",
    ]
    assert len(parameters) == 6
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in parameters)


@pytest.mark.parametrize(
    "changed",
    [
        {"branch": "reserve"},
        {"deck_id": "other-deck"},
        {"candidate_id": "other-candidate"},
    ],
)
def test_resume_identity_rejects_cross_stream_checkpoint(changed) -> None:
    from scripts.train_top2_mcts import validate_resume_identity

    expected = {"branch": "primary", "deck_id": "deck-a", "candidate_id": "candidate-a"}
    checkpoint = {**expected, **changed}

    with pytest.raises(ValueError, match="identity"):
        validate_resume_identity(checkpoint, expected)


def test_resume_identity_accepts_exact_stream() -> None:
    from scripts.train_top2_mcts import validate_resume_identity

    identity = {"branch": "primary", "deck_id": "deck-a", "candidate_id": "candidate-a"}

    validate_resume_identity(dict(identity), identity)


def test_evaluate_mcts_rows_uses_holdout_without_gradients() -> None:
    from src.rl.mcts_train import evaluate_mcts_rows

    class HoldoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            logits = batch["option_features"][..., 0] * self.scale
            return {"policy_logits": logits, "value": torch.zeros(logits.shape[0])}

    rows = [
        {
            "global_features": [0.0],
            "option_features": [[1.0], [0.0]],
            "legal_mask": [True, True],
            "player_deck": [1],
            "actions": [[0], [1]],
            "policy_target": [1.0, 0.0],
            "value_target": 0.0,
        }
    ]
    model = HoldoutModel()
    metrics = evaluate_mcts_rows(
        model,
        HoldoutModel(),
        rows,
        device=torch.device("cpu"),
        batch_size=1,
        value_coef=1.0,
        kl_coef=0.02,
        entropy_coef=0.0,
    )

    assert metrics["policy_loss"] > 0.0
    assert metrics["value_loss"] == pytest.approx(0.0)
    assert model.scale.grad is None
