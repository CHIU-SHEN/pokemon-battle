from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch


def _row(split: str) -> dict:
    return {
        "schema_version": "top2_mcts_sample_v1",
        "branch": "primary",
        "deck_id": "top2-primary-crustle-kangaskhan-cage-v1",
        "split": split,
        "global_features": [0.0] * 27,
        "option_features": [[0.0] * 80, [0.1] * 80],
        "legal_mask": [True, True],
        "player_deck": [0] * 60,
        "actions": [[0], [1]],
        "visit_counts": [3, 1],
        "policy_target": [0.75, 0.25],
        "value_target": 1.0,
    }


def test_cpu_smoke_trains_and_resumes_teacher(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    frozen_root = Path(__import__("os").environ["PTCG_FROZEN_SOURCE_ROOT"])
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "game_000000.json").write_text(
        json.dumps(
            {
                "schema_version": "top2_mcts_game_v1",
                "samples": [_row("train"), _row("valid")],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = tmp_path / "smoke.json"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_mcts_teacher_smoke.py"),
            "--project-root",
            str(frozen_root),
            "--samples",
            str(samples),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=root,
        check=True,
    )

    document = json.loads(report.read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert document["schema_version"] == "mcts_teacher_smoke_v1"
    assert document["passed"] is True
    assert document["resume_verified"] is True
    assert checkpoint["schema_version"] == "top2_mcts_teacher_checkpoint_v2"
    assert checkpoint["epoch"] == 2
    assert checkpoint["trainable_parameter_names"]
