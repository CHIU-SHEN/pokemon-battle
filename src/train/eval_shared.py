#!/usr/bin/env python3
"""Evaluate an SL-0-shared checkpoint on a frozen dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.shared_data import TrainingJsonlDataset, collate_training_rows, move_batch  # noqa: E402
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/training_manifest_v1.json")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--verify-data-sha256", action="store_true", help="Hash the full JSONL (slow for the 5 GiB dataset).")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sl0_shared_test.json")
    return parser.parse_args(argv)


class Totals:
    def __init__(self) -> None:
        self.rows = 0
        self.policy_weight = 0.0
        self.policy_loss_sum = 0.0
        self.policy_correct = 0
        self.policy_count = 0
        self.value_weight = 0.0
        self.value_loss_sum = 0.0
        self.value_squared_error = 0.0
        self.value_count = 0

    def add(self, *, policy_loss: float, policy_weight: float, policy_hit: bool | None,
            value_loss: float, value_weight: float, value_squared_error: float | None) -> None:
        self.rows += 1
        self.policy_loss_sum += policy_loss
        self.policy_weight += policy_weight
        if policy_hit is not None:
            self.policy_count += 1
            self.policy_correct += int(policy_hit)
        self.value_loss_sum += value_loss
        self.value_weight += value_weight
        if value_squared_error is not None:
            self.value_count += 1
            self.value_squared_error += value_squared_error

    def report(self) -> dict[str, int | float | None]:
        policy_loss = self.policy_loss_sum / self.policy_weight if self.policy_weight else None
        value_loss = self.value_loss_sum / self.value_weight if self.value_weight else None
        return {
            "rows": self.rows,
            "policy_loss": policy_loss,
            "policy_top1": self.policy_correct / self.policy_count if self.policy_count else None,
            "policy_count": self.policy_count,
            "value_loss": value_loss,
            "value_mse": self.value_squared_error / self.value_count if self.value_count else None,
            "value_count": self.value_count,
            "loss": (policy_loss + value_loss) if policy_loss is not None and value_loss is not None else None,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0:
        raise ValueError("batch-size must be positive; worker/sample limits cannot be negative")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("ok"):
        raise ValueError("training manifest is not valid")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "sl0_shared_checkpoint_v1":
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.get('schema_version')!r}")
    checkpoint_sha = str(checkpoint.get("dataset_sha256", "")).upper()
    manifest_sha = str(manifest.get("sha256", "")).upper()
    if not checkpoint_sha or checkpoint_sha != manifest_sha:
        raise ValueError(f"dataset hash mismatch: checkpoint={checkpoint_sha} manifest={manifest_sha}")
    actual_sha = None
    if args.verify_data_sha256:
        actual_sha = file_sha256(args.data)
        if actual_sha != manifest_sha:
            raise ValueError(f"data hash mismatch: actual={actual_sha} manifest={manifest_sha}")

    config = SharedModelConfig(**checkpoint["model_config"])
    model = SharedPolicyValueNet(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = choose_device(args.device)
    model.to(device).eval()
    amp_enabled = device.type == "cuda" and not args.no_amp
    dataset = TrainingJsonlDataset(args.data, args.split, max_samples=args.max_samples)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_training_rows, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    overall, non_forced = Totals(), Totals()
    by_source: dict[str, Totals] = {}
    illegal_predictions = 0
    batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_rows in loader:
            # Dataset metadata is intentionally retained outside collate; recover it
            # in the same streaming order through fields added below.
            sources = raw_rows.pop("policy_sources")
            forced = raw_rows.pop("forced_single_option")
            batch = move_batch(raw_rows, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - tick
            batches += 1

            logits = outputs["policy_logits"].float()
            predicted = logits.argmax(dim=-1)
            illegal_predictions += int((~batch["legal_mask"].gather(1, predicted[:, None]).squeeze(1)).sum().item())
            log_probs = F.log_softmax(logits, dim=-1)
            per_policy = -(batch["soft_policy"].float() * log_probs).sum(dim=-1)
            per_value = (outputs["value"].float() - batch["value_target"].float()).square()
            target_mask = batch["soft_policy"] > 0
            for index, source in enumerate(sources):
                pw = float(batch["policy_weight"][index].item())
                vw = float(batch["value_weight"][index].item())
                hit = bool(target_mask[index, predicted[index]].item()) if pw > 0 else None
                vse = float(per_value[index].item()) if vw > 0 else None
                values = dict(
                    policy_loss=float(per_policy[index].item()) * pw,
                    policy_weight=pw,
                    policy_hit=hit,
                    value_loss=float(per_value[index].item()) * vw,
                    value_weight=vw,
                    value_squared_error=vse,
                )
                overall.add(**values)
                by_source.setdefault(source, Totals()).add(**values)
                if not forced[index]:
                    non_forced.add(**values)

    wall_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema_version": "sl0_shared_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_best_valid_loss": checkpoint.get("best_valid_loss"),
        "split": args.split,
        "dataset_sha256": manifest_sha,
        "actual_data_sha256": actual_sha,
        "hash_verified_against_manifest": True,
        "full_file_hash_verified": actual_sha is not None,
        "device": str(device),
        "model_config": checkpoint["model_config"],
        "overall": overall.report(),
        "non_forced": non_forced.report(),
        "by_policy_source": {key: value.report() for key, value in sorted(by_source.items())},
        "legality": {"illegal_top1_predictions": illegal_predictions},
        "performance": {
            "batches": batches,
            "wall_seconds": wall_seconds,
            "inference_seconds": inference_seconds,
            "samples_per_second_wall": overall.rows / wall_seconds if wall_seconds else None,
            "samples_per_second_inference": overall.rows / inference_seconds if inference_seconds else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
