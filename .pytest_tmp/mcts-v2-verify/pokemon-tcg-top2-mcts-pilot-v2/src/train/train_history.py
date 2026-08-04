#!/usr/bin/env python3
"""Train SL-0-history with safe defaults and a frozen SL-0 warm start."""

from __future__ import annotations

import sys

from src.train.train_shared import ROOT, main as train_main


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    defaults = [
        "--model-kind", "history",
        "--data", str(ROOT / "data/training/training_decisions_history_v1.jsonl"),
        "--manifest", str(ROOT / "data/training/training_history_manifest_v1.json"),
        "--output", str(ROOT / "artifacts/sl0_history"),
    ]
    # A resumed run already contains its initialization state.  Supplying both
    # flags is intentionally rejected by train_shared, so only add the SL-0
    # warm start for a fresh run.
    if "--resume" not in user_args:
        defaults.extend([
            "--init-checkpoint", str(ROOT / "artifacts/sl0_shared_full/best.pt"),
        ])
    return train_main(defaults + user_args)


if __name__ == "__main__":
    raise SystemExit(main())
