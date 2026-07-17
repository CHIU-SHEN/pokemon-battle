#!/usr/bin/env python3
"""Evaluate an SL-1 GRU checkpoint on frozen endpoint windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.sequence_model import SequenceModelConfig, SequencePolicyValueNet  # noqa: E402
from src.train.train_sequence import loader, portable_checkpoint_paths, run_epoch  # noqa: E402
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
    model = SequencePolicyValueNet(SequenceModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_epoch(
        model, loader(args, args.split), device, None, None,
        device.type == "cuda" and not args.no_amp,
    )
    report = {
        "schema_version": "sl1_gru_eval_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "window_length": args.window_length,
        "sequence_manifest_sha256": manifest.get("output_sha256"),
        "metrics": metrics,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
