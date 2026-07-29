"""Contract tests for the portable Top2 reinforcement-learning handoff."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import torch


def test_stable_game_split_freezes_twenty_percent() -> None:
    from src.rl.top2_rollout import stable_game_split

    actual = [stable_game_split(f"game-{index}") for index in range(1000)]
    assert actual == [stable_game_split(f"game-{index}") for index in range(1000)]
    assert set(actual) == {"train", "valid", "test"}
    held_out = sum(split != "train" for split in actual)
    assert 150 <= held_out <= 250


def test_gae_propagates_terminal_reward_without_crossing_games() -> None:
    from src.rl.top2_ppo import generalized_advantage_estimate

    advantages, returns = generalized_advantage_estimate(
        rewards=[0.0, 1.0],
        values=[0.5, 0.25],
        dones=[False, True],
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert advantages == [0.5, 0.75]
    assert returns == [1.0, 1.0]


def test_masked_ppo_loss_is_finite_and_ignores_illegal_options() -> None:
    from src.rl.top2_ppo import masked_ppo_loss

    result = masked_ppo_loss(
        logits=torch.tensor([[0.0, 0.0, 1000.0]]),
        values=torch.tensor([0.0]),
        reference_logits=torch.tensor([[0.0, 0.0, -1000.0]]),
        actions=torch.tensor([0]),
        old_log_probs=torch.tensor([math.log(0.5)]),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([0.5]),
        legal_mask=torch.tensor([[True, True, False]]),
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        kl_coef=0.1,
    )
    assert all(torch.isfinite(value) for value in result.values())
    assert abs(float(result["approx_kl"])) < 1e-6
    assert abs(float(result["policy_loss"]) + 1.0) < 1e-6


def test_training_rows_reject_holdout_and_cross_deck_samples() -> None:
    from src.rl.top2_ppo import validate_training_rows

    good = {"split": "train", "deck_id": "primary-v1"}
    assert validate_training_rows([good], "primary-v1") == [good]
    for bad in (
        {"split": "valid", "deck_id": "primary-v1"},
        {"split": "test", "deck_id": "primary-v1"},
        {"split": "train", "deck_id": "reserve-v1"},
    ):
        try:
            validate_training_rows([bad], "primary-v1")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe sample was accepted: {bad}")


def test_finalize_trajectory_binds_branch_and_terminal_outcome() -> None:
    from src.rl.top2_rollout import finalize_trajectory

    rows = finalize_trajectory(
        [{"value": 0.5}, {"value": 0.25}],
        game_id="game-42",
        deck_id="primary-v1",
        result=0,
        learner_side=0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert [row["reward"] for row in rows] == [0.0, 1.0]
    assert [row["return"] for row in rows] == [1.0, 1.0]
    assert all(row["deck_id"] == "primary-v1" for row in rows)
    assert all(row["split"] == rows[0]["split"] for row in rows)
    assert rows[-1]["done"] is True


def test_verifier_accepts_only_two_hash_bound_isolated_branches() -> None:
    from scripts.verify_top2_rl_handoff import verify_policy_root

    with tempfile.TemporaryDirectory(prefix="top2_rl_verify_") as tmp:
        root = Path(tmp)
        files = {
            "artifacts/base.pt": b"base",
            "decks/primary.csv": b"1\n",
            "adapters/primary.pt": b"primary",
            "decks/reserve.csv": b"2\n",
            "adapters/reserve.pt": b"reserve",
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        digest = lambda relative: hashlib.sha256(files[relative]).hexdigest()
        policy = {
            "schema_version": "top2_rl_policy_v1",
            "release_submission_unchanged": True,
            "shared_checkpoint": {"path": "artifacts/base.pt", "sha256": digest("artifacts/base.pt")},
            "branches": [
                {"role": "primary", "candidate_id": "a", "deck_id": "a-v1", "deck_path": "decks/primary.csv", "deck_sha256": digest("decks/primary.csv"), "adapter_path": "adapters/primary.pt", "adapter_sha256": digest("adapters/primary.pt"), "budget_weight": 1.0},
                {"role": "reserve", "candidate_id": "b", "deck_id": "b-v1", "deck_path": "decks/reserve.csv", "deck_sha256": digest("decks/reserve.csv"), "adapter_path": "adapters/reserve.pt", "adapter_sha256": digest("adapters/reserve.pt"), "budget_weight": 0.4},
            ],
            "rollout": {"smoke_games_per_branch": 100, "swap_seats": True, "engine_seed_controlled": False},
            "split": {"train_percent": 80, "valid_percent": 10, "test_percent": 10, "holdout_never_train_percent": 20},
            "ppo": {"algorithm": "masked_ppo", "kl_coef": 0.05},
            "pilot": {
                "safety_gates": {"target_kl_max": 0.03, "clip_fraction_max": 0.3, "entropy_drop_max": 0.5},
                "presets": [
                    {"name": "conservative", "learning_rate": 0.00005, "clip_ratio": 0.10, "kl_coef": 0.10, "entropy_coef": 0.005, "epochs": 3},
                    {"name": "baseline", "learning_rate": 0.00010, "clip_ratio": 0.15, "kl_coef": 0.05, "entropy_coef": 0.010, "epochs": 4},
                    {"name": "exploratory", "learning_rate": 0.00020, "clip_ratio": 0.20, "kl_coef": 0.02, "entropy_coef": 0.020, "epochs": 4},
                ],
            },
            "release_gates": {"submission_replacement_authorized": False},
        }
        (root / "config").mkdir()
        (root / "config/top2_rl_policy.json").write_text(json.dumps(policy), encoding="utf-8")
        report = verify_policy_root(root)
        assert report["roles"] == ["primary", "reserve"]
        assert report["verified_hashes"] == 5
        policy["branches"][1]["deck_id"] = "a-v1"
        (root / "config/top2_rl_policy.json").write_text(json.dumps(policy), encoding="utf-8")
        try:
            verify_policy_root(root)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate deck_id was accepted")


def test_archive_writer_creates_single_root_manifest_and_sidecar() -> None:
    from scripts.build_top2_rl_handoff import PACKAGE_BASENAME, write_handoff_archive

    with tempfile.TemporaryDirectory(prefix="top2_rl_archive_") as tmp:
        base = Path(tmp)
        package_root = base / PACKAGE_BASENAME
        package_root.mkdir()
        (package_root / "payload.txt").write_bytes(b"rl-handoff\n")
        archive, sidecar, manifest = write_handoff_archive(package_root, base / "output")
        assert archive.is_file() and sidecar.is_file()
        assert manifest["sha256"] == {
            "payload.txt": hashlib.sha256(b"rl-handoff\n").hexdigest()
        }
        assert sidecar.read_text(encoding="utf-8").endswith(f"  {archive.name}\n")
        with tarfile.open(archive, "r:gz") as stream:
            roots = {Path(name).parts[0] for name in stream.getnames() if name}
        assert roots == {PACKAGE_BASENAME}


def test_v1_queue_keeps_only_train_samples_from_one_deck() -> None:
    from scripts.select_top2_v1_candidates import rank_candidates

    rows = [
        {"sample_id": "loss", "deck_id": "primary-v1", "split": "train", "game_result": "loss", "confidence": 0.3, "entropy": 1.0},
        {"sample_id": "low", "deck_id": "primary-v1", "split": "train", "game_result": "win", "confidence": 0.4, "entropy": 0.8},
        {"sample_id": "holdout", "deck_id": "primary-v1", "split": "test", "game_result": "loss", "confidence": 0.1, "entropy": 2.0},
        {"sample_id": "cross", "deck_id": "reserve-v1", "split": "train", "game_result": "loss", "confidence": 0.1, "entropy": 2.0},
    ]
    selected = rank_candidates(rows, deck_id="primary-v1", max_items=10)
    assert [row["sample_id"] for row in selected] == ["loss", "low"]


def test_pilot_budget_tiers_preserve_rollout_and_degrade_evaluation() -> None:
    from src.rl.pilot import choose_budget_tier

    full = choose_budget_tier(5400.0)
    reduced = choose_budget_tier(7200.0)
    minimal = choose_budget_tier(7200.1)
    assert (full.name, full.rollout_games_per_branch, full.arena_games, full.trials) == (
        "full", 100, 200, ("conservative", "baseline", "exploratory")
    )
    assert (reduced.name, reduced.rollout_games_per_branch, reduced.arena_games) == ("reduced", 100, 100)
    assert (minimal.name, minimal.rollout_games_per_branch, minimal.arena_games, minimal.epoch_cap) == ("minimal", 100, 0, 2)


def test_wilson_interval_and_preliminary_tie_break_are_deterministic() -> None:
    from src.rl.pilot import select_preliminary_trial, wilson_interval

    low, high = wilson_interval(50, 100)
    assert abs(low - 0.4038315) < 1e-6
    assert abs(high - 0.5961685) < 1e-6
    overlapping = [
        {"name": "exploratory", "eligible": False, "arena_wins": 180, "arena_games": 200},
        {"name": "baseline", "eligible": True, "arena_wins": 120, "arena_games": 200},
        {"name": "conservative", "eligible": True, "arena_wins": 110, "arena_games": 200},
    ]
    result = select_preliminary_trial(overlapping)
    assert result["selected"] == "conservative"
    assert result["status"] == "preliminary_intervals_overlap"
    separated = [
        {"name": "baseline", "eligible": True, "arena_wins": 160, "arena_games": 200},
        {"name": "conservative", "eligible": True, "arena_wins": 90, "arena_games": 200},
    ]
    assert select_preliminary_trial(separated)["selected"] == "baseline"


def test_ppo_safety_stop_reason_catches_each_hard_gate() -> None:
    from src.rl.pilot import safety_stop_reason

    base = {
        "loss": 0.2,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "approx_kl": 0.01,
        "clip_fraction": 0.1,
    }
    limits = {"target_kl_max": 0.03, "clip_fraction_max": 0.30, "entropy_drop_max": 0.50}
    assert safety_stop_reason(base, first_entropy=1.0, limits=limits) is None
    for key, value, expected in (
        ("loss", float("nan"), "non_finite"),
        ("approx_kl", 0.031, "kl_limit"),
        ("clip_fraction", 0.31, "clip_fraction_limit"),
        ("entropy", 0.49, "entropy_collapse"),
    ):
        metrics = dict(base)
        metrics[key] = value
        assert safety_stop_reason(metrics, first_entropy=1.0, limits=limits) == expected


def main() -> int:
    test_stable_game_split_freezes_twenty_percent()
    test_gae_propagates_terminal_reward_without_crossing_games()
    test_masked_ppo_loss_is_finite_and_ignores_illegal_options()
    test_training_rows_reject_holdout_and_cross_deck_samples()
    test_finalize_trajectory_binds_branch_and_terminal_outcome()
    test_verifier_accepts_only_two_hash_bound_isolated_branches()
    test_archive_writer_creates_single_root_manifest_and_sidecar()
    test_v1_queue_keeps_only_train_samples_from_one_deck()
    test_pilot_budget_tiers_preserve_rollout_and_degrade_evaluation()
    test_wilson_interval_and_preliminary_tie_break_are_deterministic()
    test_ppo_safety_stop_reason_catches_each_hard_gate()
    print("OK: Top2 RL handoff contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
