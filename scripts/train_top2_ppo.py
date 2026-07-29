#!/usr/bin/env python3
"""Train one branch-bound Top2 Adapter with masked PPO."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.top2_ppo import collate_rollout_rows, load_rollout_rows, masked_ppo_loss  # noqa: E402
from src.rl.top2_rollout import Top2RolloutAgent, sha256_file  # noqa: E402
from src.train.shared_data import move_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    owner = Top2RolloutAgent(
        branch["candidate_id"], branch["deck_id"], project_root=args.project_root.resolve(), device=str(device), deterministic=True, record_decisions=False
    )
    model = owner.model.to(device)
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    rows = load_rollout_rows(args.rollouts.resolve(), branch["deck_id"])
    advantage_tensor = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32)
    advantage_mean = float(advantage_tensor.mean())
    advantage_std = float(advantage_tensor.std(unbiased=False).clamp_min(1e-6))
    for row in rows:
        row["advantage"] = (float(row["advantage"]) - advantage_mean) / advantage_std
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    ppo = config["ppo"]
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate_rollout_rows)
        totals = {key: 0.0 for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        batches = 0
        model.train()
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            with torch.no_grad():
                reference_output = reference(batch)
            losses = masked_ppo_loss(
                logits=output["policy_logits"], values=output["value"], reference_logits=reference_output["policy_logits"],
                actions=batch["actions"], old_log_probs=batch["old_log_probs"], advantages=batch["advantages"], returns=batch["returns"],
                legal_mask=batch["legal_mask"], clip_ratio=float(ppo["clip_ratio"]), value_coef=float(ppo["value_coef"]),
                entropy_coef=float(ppo["entropy_coef"]), kl_coef=float(ppo["kl_coef"]),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(ppo["max_grad_norm"]))
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        metrics = {key: value / max(batches, 1) for key, value in totals.items()}
        metrics.update({"epoch": epoch, "batches": batches})
        history.append(metrics)
        payload = {
            "schema_version": "top2_ppo_checkpoint_v1", "candidate_id": branch["candidate_id"], "deck_id": branch["deck_id"],
            "initial_adapter_sha256": sha256_file(owner.adapter_path), "adapter_state": model.adapter_state_dict(),
            "epoch": epoch, "metrics": metrics, "ppo_config": ppo,
        }
        torch.save(payload, args.output / "last.pt")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    (args.output / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
