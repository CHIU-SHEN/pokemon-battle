#!/usr/bin/env python3
"""M6 distillation pipeline smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="m6_distill_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "samples.jsonl"
        schema_path = tmp_path / "schema.json"
        summary_path = tmp_path / "collect_summary.json"
        model_path = tmp_path / "model.json"
        metrics_path = tmp_path / "metrics.json"
        splits_path = tmp_path / "splits.json"
        queue_path = tmp_path / "queue.json"

        run(
            [
                sys.executable,
                "src/train/collect_distill.py",
                "--max-samples",
                "20",
                "--group-size",
                "5",
                "--search",
                "--max-candidates",
                "3",
                "--particles",
                "1",
                "--node-budget",
                "8",
                "--time-budget",
                "0.01",
                "--out",
                str(data_path),
                "--schema-out",
                str(schema_path),
                "--summary-out",
                str(summary_path),
            ]
        )
        run(
            [
                sys.executable,
                "src/train/train_distill.py",
                "--data",
                str(data_path),
                "--model-out",
                str(model_path),
                "--metrics-out",
                str(metrics_path),
                "--splits-out",
                str(splits_path),
                "--epochs",
                "12",
            ]
        )
        run(
            [
                sys.executable,
                "src/train/reanalysis.py",
                "--data",
                str(data_path),
                "--model",
                str(model_path),
                "--out",
                str(queue_path),
                "--max-items",
                "10",
            ]
        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = json.loads(model_path.read_text(encoding="utf-8"))
        queue = json.loads(queue_path.read_text(encoding="utf-8"))

        assert summary["ok"], summary
        assert summary["sample_count"] == 20
        assert len(model["policy_weights"]) == len(model["feature_names"])
        assert model["status"] == "experimental_not_promoted"
        assert metrics["split"]["train_samples"] > 0
        assert metrics["latency_ms_per_select"] is None or metrics["latency_ms_per_select"] < 1.0
        assert queue["items"]

    print("OK: M6 distillation collection, training, and reanalysis pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

