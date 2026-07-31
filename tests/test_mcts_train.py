from __future__ import annotations

import pytest
import torch


def test_mcts_loss_uses_soft_visit_target_and_terminal_value() -> None:
    from src.rl.mcts_train import mcts_loss

    result = mcts_loss(
        logits=torch.tensor([[2.0, 0.0]], requires_grad=True),
        values=torch.tensor([0.2], requires_grad=True),
        reference_logits=torch.tensor([[1.5, 0.5]]),
        actions=[[(0,), (1,)]],
        policy_targets=[[0.75, 0.25]],
        value_targets=torch.tensor([1.0]),
        legal_mask=torch.tensor([[True, True]]),
        value_coef=1.0,
        kl_coef=0.02,
        entropy_coef=0.005,
    )

    assert result["loss"].isfinite()
    assert result["policy_loss"].item() > 0.0
    assert result["value_loss"].item() == pytest.approx(0.64)
    result["loss"].backward()


def test_mcts_loss_masks_illegal_options() -> None:
    from src.rl.mcts_train import mcts_loss

    result = mcts_loss(
        logits=torch.tensor([[0.0, 100.0]]),
        values=torch.tensor([0.0]),
        reference_logits=torch.tensor([[0.0, 100.0]]),
        actions=[[(0,)]],
        policy_targets=[[1.0]],
        value_targets=torch.tensor([0.0]),
        legal_mask=torch.tensor([[True, False]]),
        value_coef=1.0,
        kl_coef=0.02,
        entropy_coef=0.0,
    )

    assert result["policy_loss"].item() == pytest.approx(0.0)


def test_loader_rejects_cross_branch_and_non_train_samples(tmp_path) -> None:
    from src.rl.mcts_train import load_mcts_rows

    path = tmp_path / "game_000000.json"
    path.write_text(
        """
{"schema_version":"top2_mcts_game_v1","samples":[
 {"schema_version":"top2_mcts_sample_v1","branch":"reserve","deck_id":"d","split":"train"},
 {"schema_version":"top2_mcts_sample_v1","branch":"primary","deck_id":"d","split":"valid"}
]}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="branch"):
        load_mcts_rows(tmp_path, branch="primary", deck_id="d", split="train")
