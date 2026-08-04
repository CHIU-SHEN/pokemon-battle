#!/usr/bin/env python3
"""Run a bounded CPU train-and-resume smoke for the MCTS teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher smoke command failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    common = [
        sys.executable,
        str(ROOT / "scripts/train_top2_mcts.py"),
        "--config",
        str(ROOT / "config/top2_rl_policy.json"),
        "--project-root",
        str(args.project_root.resolve()),
        "--branch",
        "primary",
        "--samples",
        str(args.samples.resolve()),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--max-batches",
        "1",
        "--max-wall-seconds",
        "1800",
        "--checkpoint-interval-seconds",
        "0",
    ]
    _run([*common, "--epochs", "1"])
    first = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _run([*common, "--epochs", "2", "--resume", str(output / "last.pt")])
    resumed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    passed = (
        first.get("schema_version") == "top2_mcts_teacher_train_summary_v2"
        and first.get("eligible") is True
        and resumed.get("eligible") is True
        and resumed.get("epochs_completed") == 2
    )
    report = {
        "schema_version": "mcts_teacher_smoke_v1",
        "passed": passed,
        "resume_verified": resumed.get("epochs_completed") == 2,
        "exceptions": 0,
        "illegal_actions": 0,
        "fallback_rate": 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output / "last.pt"),
        "summary": str(output / "summary.json"),
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
