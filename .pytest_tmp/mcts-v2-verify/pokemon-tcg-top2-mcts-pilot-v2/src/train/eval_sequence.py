#!/usr/bin/env python3
"""Evaluate an SL-1 GRU checkpoint on frozen endpoint windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.sequence_model import SequenceModelConfig, SequencePolicyValueNet  # noqa: E402
from src.train.eval_shared import Totals  # noqa: E402
from src.train.shared_data import move_batch  # noqa: E402
from src.train.train_sequence import loader, portable_checkpoint_paths  # noqa: E402
from src.train.train_shared import choose_device  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data/training/sequence_trajectories_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/sequence_manifest_v1.json")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Reuse the loader's validation cap slot for either non-training split.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with portable_checkpoint_paths():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "sl1_gru_checkpoint_v1":
        raise ValueError("checkpoint is not SL-1 GRU")
    if checkpoint.get("sequence_manifest_sha256") != manifest.get("output_sha256"):
        raise ValueError("checkpoint and sequence index hashes do not match")
    if checkpoint.get("dataset_sha256") != manifest.get("source_sha256"):
        raise ValueError("checkpoint and source dataset hashes do not match")
    device = choose_device(args.device, 0)
    model = SequencePolicyValueNet(SequenceModelConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    amp_enabled = device.type == "cuda" and not args.no_amp
    overall, non_forced = Totals(), Totals()
    by_source: dict[str, Totals] = {}
    illegal_predictions = 0
    batches = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader(args, args.split):
            endpoint_indices = batch["endpoint_flat_indices"].tolist()
            flat = batch["flat_batch"]
            sources = [flat["policy_sources"][index] for index in endpoint_indices]
            forced = [flat["forced_single_option"][index] for index in endpoint_indices]
            batch = move_batch(batch, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - tick
            batches += 1

            indices = batch["endpoint_flat_indices"]
            targets = batch["flat_batch"]
            legal_mask = targets["legal_mask"][indices]
            soft_policy = targets["soft_policy"][indices]
            policy_weight = targets["policy_weight"][indices]
            value_target = targets["value_target"][indices]
            value_weight = targets["value_weight"][indices]
            logits = outputs["policy_logits"].float()
            predicted = logits.argmax(dim=-1)
            illegal_predictions += int(
                (~legal_mask.gather(1, predicted[:, None]).squeeze(1)).sum().item()
            )
            per_policy = -(soft_policy.float() * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            per_value = (outputs["value"].float() - value_target.float()).square()
            target_mask = soft_policy > 0
            for index, source in enumerate(sources):
                pw = float(policy_weight[index].item())
                vw = float(value_weight[index].item())
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
        "schema_version": "sl1_gru_eval_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_valid_loss": checkpoint.get("best_valid_loss"),
        "split": args.split,
        "window_length": args.window_length,
        "sequence_manifest_sha256": manifest.get("output_sha256"),
        "dataset_sha256": manifest.get("source_sha256"),
        "model_config": checkpoint["model_config"],
        "overall": overall.report(),
        "non_forced": non_forced.report(),
        "by_policy_source": {
            key: value.report() for key, value in sorted(by_source.items())
        },
        "legality": {"illegal_top1_predictions": illegal_predictions},
        "performance": {
            "batches": batches,
            "wall_seconds": wall_seconds,
            "inference_seconds": inference_seconds,
            "samples_per_second_wall": overall.rows / wall_seconds if wall_seconds else None,
            "samples_per_second_inference": (
                overall.rows / inference_seconds if inference_seconds else None
            ),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
