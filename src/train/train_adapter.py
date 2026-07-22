#!/usr/bin/env python3
"""Train one lightweight deck Adapter on a frozen SL-0-shared checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.adapter_data import AdapterJsonlDataset
from src.train.adapter_model import DeckAdapterPolicyValueNet
from src.train.shared_data import collate_training_rows, move_batch
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, batch_metrics, weighted_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True, help="Repeat for base and supplemental JSONL files.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-test cap; 0 means full split.")
    return parser.parse_args()


def load_base(path: Path, device: torch.device) -> tuple[SharedPolicyValueNet, dict]:
    import pathlib
    original_posix = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix  # type: ignore[misc,assignment]
    base = SharedPolicyValueNet(SharedModelConfig(**checkpoint["model_config"]))
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval().to(device)
    return base, checkpoint


def run_epoch(model, loader, device, optimizer=None, max_batches: int = 0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "correct": 0.0, "policy_n": 0.0, "value_se": 0.0, "value_n": 0.0, "batches": 0}
    for index, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            losses = weighted_losses(outputs, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
        metrics = batch_metrics(outputs, batch)
        for key in ("loss", "policy_loss", "value_loss"):
            totals[key] += float(losses[key].item())
        totals["correct"] += metrics["policy_correct"]
        totals["policy_n"] += metrics["policy_count"]
        totals["value_se"] += metrics["value_squared_error"]
        totals["value_n"] += metrics["value_count"]
        totals["batches"] += 1
        if max_batches and index >= max_batches:
            break
    batches = max(totals["batches"], 1)
    return {"loss": totals["loss"] / batches, "policy_loss": totals["policy_loss"] / batches, "value_loss": totals["value_loss"] / batches, "policy_top1": totals["correct"] / max(totals["policy_n"], 1), "value_mse": totals["value_se"] / max(totals["value_n"], 1), "batches": totals["batches"]}


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    base, base_checkpoint = load_base(args.base_checkpoint, device)
    model = DeckAdapterPolicyValueNet(base, args.bottleneck_dim).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        loaders = {}
        for split in ("train", "valid"):
            dataset = AdapterJsonlDataset(args.data, args.view, split)
            loaders[split] = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_training_rows)
        train = run_epoch(model, loaders["train"], device, optimizer, args.max_batches)
        valid = run_epoch(model, loaders["valid"], device, None, args.max_batches)
        record = {"epoch": epoch, "train": train, "valid": valid}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        payload = {"schema_version": "deck_adapter_checkpoint_v1", "candidate_id": json.loads(args.view.read_text(encoding="utf-8"))["candidate_id"], "base_dataset_sha256": base_checkpoint.get("dataset_sha256"), "bottleneck_dim": args.bottleneck_dim, "adapter_state": model.adapter_state_dict(), "epoch": epoch, "metrics": record}
        torch.save(payload, args.output / "last.pt")
        if valid["loss"] < best:
            best = valid["loss"]
            torch.save(payload, args.output / "best.pt")
    (args.output / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
