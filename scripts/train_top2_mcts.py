#!/usr/bin/env python3
"""Train a Top2 adapter from belief-PUCT visit targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.rl.mcts_train import collate_mcts_rows, load_mcts_rows, mcts_loss  # noqa: E402
from src.train.shared_data import move_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def load_adapter_state(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") not in {
        "top2_ppo_checkpoint_v1",
        "top2_mcts_checkpoint_v1",
    }:
        raise ValueError("unsupported initial checkpoint")
    state = checkpoint["adapter_state"]
    model.adapter.load_state_dict(state["adapter"], strict=True)
    model.policy_delta.load_state_dict(state["policy_delta"], strict=True)
    model.value_delta.load_state_dict(state["value_delta"], strict=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    owner = AdapterArenaAgent(
        branch["candidate_id"],
        project_root=args.project_root.resolve(),
        device=device,
    )
    model = owner.model.to(device)
    if args.initial_checkpoint:
        load_adapter_state(model, args.initial_checkpoint.resolve())
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    train_rows = load_mcts_rows(
        args.samples.resolve(),
        branch=args.branch,
        deck_id=branch["deck_id"],
        split="train",
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_rows)
        sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reference_kl": 0.0, "entropy": 0.0}
        batches = 0
        model.train()
        for offset in range(0, len(train_rows), args.batch_size):
            raw = collate_mcts_rows(train_rows[offset: offset + args.batch_size])
            actions = raw.pop("actions")
            policy_targets = raw.pop("policy_targets")
            value_targets = raw.pop("value_targets").to(device)
            batch = move_batch(raw, device)
            output_doc = model(batch)
            with torch.no_grad():
                reference_doc = reference(batch)
            loss_doc = mcts_loss(
                logits=output_doc["policy_logits"],
                values=output_doc["value"],
                reference_logits=reference_doc["policy_logits"],
                actions=actions,
                policy_targets=policy_targets,
                value_targets=value_targets,
                legal_mask=batch["legal_mask"],
                value_coef=1.0,
                kl_coef=0.02,
                entropy_coef=0.005,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_doc["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            for key in sums:
                sums[key] += float(loss_doc[key].detach().item())
            batches += 1
        record = {
            "epoch": epoch,
            **{key: value / max(1, batches) for key, value in sums.items()},
            "batches": batches,
            "samples": len(train_rows),
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics.append(record)
        print(json.dumps(record), flush=True)
    final = metrics[-1]
    eligible = all(math.isfinite(float(value)) for key, value in final.items() if isinstance(value, float))
    checkpoint = {
        "schema_version": "top2_mcts_checkpoint_v1",
        "branch": args.branch,
        "candidate_id": branch["candidate_id"],
        "deck_id": branch["deck_id"],
        "adapter_state": {
            "adapter": model.adapter.state_dict(),
            "policy_delta": model.policy_delta.state_dict(),
            "value_delta": model.value_delta.state_dict(),
        },
        "metrics": final,
    }
    torch.save(checkpoint, output / "last.pt")
    summary = {
        "schema_version": "top2_mcts_train_summary_v1",
        "eligible": eligible,
        "checkpoint": str(output / "last.pt"),
        "epochs": metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
