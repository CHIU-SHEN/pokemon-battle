#!/usr/bin/env python3
"""Train SL-1 GRU on leakage-safe same-perspective decision windows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.sequence_data import SequenceWindowDataset, collate_sequence_windows  # noqa: E402
from src.train.sequence_model import (  # noqa: E402
    SequenceModelConfig,
    SequencePolicyValueNet,
    endpoint_targets,
    initialize_from_sl0,
)
from src.train.shared_data import move_batch  # noqa: E402
from src.train.shared_model import batch_metrics, weighted_losses  # noqa: E402
from src.train.train_shared import choose_device, inspect_dimensions, seed_everything  # noqa: E402
from src.train.transition_features import TRANSITION_DIM  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data/training/sequence_trajectories_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/sequence_manifest_v1.json")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "artifacts/sl0_shared_full/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sl1_gru")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--gru-hidden-dim", type=int, default=192)
    parser.add_argument("--option-hidden-dim", type=int, default=192)
    parser.add_argument("--deck-embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-card-id", type=int, default=1267)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


@contextmanager
def portable_checkpoint_paths():
    import pathlib
    original = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        pathlib.PosixPath = original  # type: ignore[misc,assignment]


def loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = SequenceWindowDataset(
        args.data,
        args.index,
        split,
        window_length=args.window_length,
        max_windows=args.max_train_windows if split == "train" else args.max_valid_windows,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_sequence_windows,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, batches, device, optimizer=None, scaler=None, amp=False, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "batches": 0,
              "policy_correct": 0.0, "policy_count": 0.0, "value_squared_error": 0.0, "value_count": 0.0}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in batches:
            batch = move_batch(batch, device)
            targets = endpoint_targets(batch)
            autocast = torch.autocast(device_type=device.type, dtype=torch.float16) if amp else nullcontext()
            with autocast:
                outputs = model(batch)
                losses = weighted_losses(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            metrics = batch_metrics(outputs, targets)
            for key in ("loss", "policy_loss", "value_loss"):
                totals[key] += float(losses[key].item())
            for key, value in metrics.items():
                totals[key] += value
            totals["batches"] += 1
    denom = max(totals["batches"], 1)
    return {
        "loss": totals["loss"] / denom,
        "policy_loss": totals["policy_loss"] / denom,
        "value_loss": totals["value_loss"] / denom,
        "policy_top1": totals["policy_correct"] / max(totals["policy_count"], 1.0),
        "value_mse": totals["value_squared_error"] / max(totals["value_count"], 1.0),
        "batches": totals["batches"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = choose_device(args.device, 0)
    seed_everything(args.seed, 0, False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("ok"):
        raise ValueError("sequence manifest is not valid")
    global_dim, option_dim = inspect_dimensions(args.data)
    config = SequenceModelConfig(
        global_dim=global_dim, option_dim=option_dim, transition_dim=TRANSITION_DIM,
        max_card_id=args.max_card_id, hidden_dim=args.hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim, option_hidden_dim=args.option_hidden_dim,
        deck_embedding_dim=args.deck_embedding_dim, dropout=args.dropout,
    )
    model = SequencePolicyValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, best = 0, float("inf")
    checkpoint_path = args.resume or args.init_checkpoint
    with portable_checkpoint_paths():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if args.resume:
        if checkpoint.get("schema_version") != "sl1_gru_checkpoint_v1":
            raise ValueError("resume checkpoint is not SL-1 GRU")
        if checkpoint.get("sequence_manifest_sha256") != manifest.get("output_sha256"):
            raise ValueError("resume checkpoint was trained with a different sequence index")
        if checkpoint.get("dataset_sha256") != manifest.get("source_sha256"):
            raise ValueError("resume checkpoint was trained with a different source dataset")
        if checkpoint.get("sequence_manifest_sha256") != manifest.get("output_sha256"):
            raise ValueError("resume checkpoint was trained with a different sequence index")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state") or {})
        start_epoch, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_valid_loss"])
    else:
        if checkpoint.get("schema_version") != "sl0_shared_checkpoint_v1":
            raise ValueError("initial checkpoint must be SL-0 shared")
        initialize_from_sl0(model, checkpoint)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, loader(args, "train"), device, optimizer, scaler, amp, args.grad_clip)
        valid_metrics = run_epoch(model, loader(args, "valid"), device, None, scaler, amp, args.grad_clip)
        document = {
            "schema_version": "sl1_gru_checkpoint_v1", "model_kind": "sequence_gru",
            "model_config": config.to_dict(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
            "epoch": epoch, "best_valid_loss": min(best, valid_metrics["loss"]),
            "sequence_manifest_sha256": manifest.get("output_sha256"),
            "dataset_sha256": manifest.get("source_sha256"),
            "dataset_sha256": manifest.get("source_sha256"),
            "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics": {"train": train_metrics, "valid": valid_metrics},
        }
        torch.save(document, args.output / "last.pt")
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            document["best_valid_loss"] = best
            torch.save(document, args.output / "best.pt")
        print(json.dumps({"epoch": epoch, "train": train_metrics, "valid": valid_metrics,
                          "elapsed_sec": time.perf_counter() - started}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
