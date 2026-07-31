#!/usr/bin/env python3
"""Evaluate a Top2 PPO checkpoint on the frozen valid/test rollout splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.top2_ppo import (  # noqa: E402
    collate_rollout_rows,
    holdout_batch_metrics,
    load_rollout_rows_for_splits,
)
from src.rl.top2_rollout import Top2RolloutAgent, sha256_file  # noqa: E402
from src.train.shared_data import move_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    rows = load_rollout_rows_for_splits(args.rollouts.resolve(), branch["deck_id"], splits)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    candidate = Top2RolloutAgent(
        branch["candidate_id"], branch["deck_id"], project_root=args.project_root.resolve(),
        device=str(device), ppo_checkpoint=args.checkpoint.resolve(), deterministic=True, record_decisions=False,
    ).model.eval()
    reference = Top2RolloutAgent(
        branch["candidate_id"], branch["deck_id"], project_root=args.project_root.resolve(),
        device=str(device),
        ppo_checkpoint=args.reference_checkpoint.resolve() if args.reference_checkpoint else None,
        deterministic=True,
        record_decisions=False,
    ).model.eval()
    totals = {
        "samples": 0, "candidate_correct": 0, "reference_correct": 0,
        "action_agreement": 0, "illegal_argmax": 0, "reference_kl_sum": 0.0,
        "candidate_value_se": 0.0, "reference_value_se": 0.0,
    }
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate_rollout_rows)
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            candidate_output = candidate(batch)
            reference_output = reference(batch)
            metrics = holdout_batch_metrics(
                candidate_logits=candidate_output["policy_logits"],
                reference_logits=reference_output["policy_logits"],
                candidate_values=candidate_output["value"],
                reference_values=reference_output["value"],
                actions=batch["actions"], returns=batch["returns"], legal_mask=batch["legal_mask"],
            )
            for key in totals:
                totals[key] += metrics[key]
    samples = max(int(totals["samples"]), 1)
    report = {
        "schema_version": "top2_ppo_holdout_v1",
        "role": args.branch,
        "candidate_id": branch["candidate_id"],
        "deck_id": branch["deck_id"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()) if args.reference_checkpoint else None,
        "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint.resolve()) if args.reference_checkpoint else None,
        "splits": sorted(splits),
        "split_samples": {split: sum(row["split"] == split for row in rows) for split in sorted(splits)},
        "games": len({row["game_id"] for row in rows}),
        "samples": int(totals["samples"]),
        "candidate_action_accuracy": totals["candidate_correct"] / samples,
        "reference_action_accuracy": totals["reference_correct"] / samples,
        "candidate_reference_action_agreement": totals["action_agreement"] / samples,
        "reference_kl": totals["reference_kl_sum"] / samples,
        "candidate_value_mse": totals["candidate_value_se"] / samples,
        "reference_value_mse": totals["reference_value_se"] / samples,
        "illegal_argmax": int(totals["illegal_argmax"]),
        "finite": all(math.isfinite(float(totals[key])) for key in ("reference_kl_sum", "candidate_value_se", "reference_value_se")),
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["finite"] and report["illegal_argmax"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
