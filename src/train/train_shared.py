#!/usr/bin/env python3
"""Train SL-0-shared on the formal dynamic-option training dataset."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.shared_data import TrainingJsonlDataset, collate_training_rows, move_batch  # noqa: E402
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet, batch_metrics, weighted_losses  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/training_manifest_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sl0_shared")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--option-hidden-dim", type=int, default=192)
    parser.add_argument("--deck-embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-card-id", type=int, default=1267)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 uses the complete train split; use num-workers=0 for an exact smoke cap.")
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args(argv)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


def choose_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        requested = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def seed_everything(seed: int, rank: int, deterministic: bool) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("ok"):
        raise ValueError(f"training manifest is not valid: {path}")
    return manifest


def inspect_dimensions(path: Path, split: str = "train") -> tuple[int, int]:
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            global_dim = len(row["features"])
            if not row["option_features"]:
                continue
            return global_dim, len(row["option_features"][0])
    raise ValueError(f"no usable {split} sample found in {path}")


def make_loader(
    args: argparse.Namespace,
    split: str,
    epoch: int,
    rank: int,
    world_size: int,
) -> tuple[TrainingJsonlDataset, DataLoader]:
    training = split == "train"
    max_samples = args.max_train_samples if training else args.max_valid_samples
    dataset = TrainingJsonlDataset(
        args.data,
        split,
        shuffle_buffer=args.shuffle_buffer if training else 0,
        seed=args.seed,
        epoch=epoch,
        max_samples=max_samples,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_training_rows,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return dataset, loader


def amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def reduce_totals(totals: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return totals


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with amp_context(device, amp_enabled):
                outputs = model(batch)
                losses = weighted_losses(outputs, batch)
            metrics = batch_metrics(outputs, batch)
            totals += torch.tensor([
                float(losses["loss"].item()),
                float(losses["policy_loss"].item()),
                float(losses["value_loss"].item()),
                1.0,
                metrics["policy_correct"],
                metrics["policy_count"],
                metrics["value_squared_error"],
                metrics["value_count"],
            ], dtype=torch.float64, device=device)
    totals = reduce_totals(totals, world_size).cpu()
    batches = max(float(totals[3]), 1.0)
    return {
        "loss": float(totals[0] / batches),
        "policy_loss": float(totals[1] / batches),
        "value_loss": float(totals[2] / batches),
        "policy_top1": float(totals[4] / max(float(totals[5]), 1.0)),
        "value_mse": float(totals[6] / max(float(totals[7]), 1.0)),
        "batches": float(totals[3]),
    }


def raw_model(model: nn.Module) -> SharedPolicyValueNet:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    epoch: int,
    global_step: int,
    best_valid_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sl0_shared_checkpoint_v1",
        "model_config": raw_model(model).config.to_dict(),
        "model_state": raw_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_valid_loss": best_valid_loss,
        "training_args": vars(args),
        "dataset_sha256": manifest.get("sha256"),
        "dataset_samples": manifest.get("samples"),
        "metrics": metrics,
    }, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("epochs, batch-size and grad-accum must be positive")
    rank, world_size, local_rank = distributed_context()
    device = choose_device(args.device, local_rank)
    seed_everything(args.seed, rank, args.deterministic)
    manifest = read_manifest(args.manifest)
    global_dim, option_dim = inspect_dimensions(args.data)
    config = SharedModelConfig(
        global_dim=global_dim,
        option_dim=option_dim,
        max_card_id=args.max_card_id,
        hidden_dim=args.hidden_dim,
        option_hidden_dim=args.option_hidden_dim,
        deck_embedding_dim=args.deck_embedding_dim,
        dropout=args.dropout,
    )
    model: nn.Module = SharedPolicyValueNet(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_valid_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("dataset_sha256") != manifest.get("sha256"):
            raise ValueError("resume checkpoint was trained on a different dataset hash")
        raw_model(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state") or {})
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_config.json").write_text(json.dumps({
            "model": config.to_dict(),
            "training": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "dataset_sha256": manifest.get("sha256"),
            "world_size": world_size,
            "device": str(device),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        _, train_loader = make_loader(args, "train", epoch, rank, world_size)
        _, valid_loader = make_loader(args, "valid", epoch, rank, world_size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        running_loss = 0.0
        batch_count = 0
        samples_seen = 0
        for batch_index, batch in enumerate(train_loader):
            samples_seen += len(batch["sample_ids"])
            batch = move_batch(batch, device)
            with amp_context(device, amp_enabled):
                outputs = model(batch)
                losses = weighted_losses(outputs, batch)
                loss = losses["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_index + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            running_loss += float(losses["loss"].item())
            batch_count += 1
            if rank == 0 and args.log_every and batch_count % args.log_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(json.dumps({
                    "epoch": epoch,
                    "batch": batch_count,
                    "global_step": global_step,
                    "loss": running_loss / batch_count,
                    "samples_per_second": samples_seen / elapsed,
                }, ensure_ascii=False), flush=True)
        # Flush a final partial accumulation rather than silently dropping it.
        if batch_count and batch_count % args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        valid = evaluate(model, valid_loader, device, amp_enabled, world_size)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(batch_count, 1),
            "train_batches": batch_count,
            "samples_seen_per_rank": samples_seen,
            "epoch_seconds": time.perf_counter() - started,
            "valid": valid,
        }
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            improved = valid["loss"] < best_valid_loss
            best_valid_loss = min(best_valid_loss, valid["loss"])
            save_checkpoint(
                args.output / "last.pt", model, optimizer, scaler, args, manifest,
                epoch, global_step, best_valid_loss, epoch_metrics,
            )
            if improved:
                save_checkpoint(
                    args.output / "best.pt", model, optimizer, scaler, args, manifest,
                    epoch, global_step, best_valid_loss, epoch_metrics,
                )
            print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
        if world_size > 1:
            torch.distributed.barrier()

    if world_size > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
