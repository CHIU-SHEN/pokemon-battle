from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _bash() -> str:
    candidates = [
        "C:/Program Files/Git/bin/bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("bash is unavailable")


def test_quality_gated_job_dry_run_exposes_safe_stage_order(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_FULL_PIPELINE": "1",
        "RUN_ROOT": str(tmp_path / "run"),
        "MCTS_WORKERS": "2",
    }

    completed = subprocess.run(
        [_bash(), "jobs/mcts_teacher_v3_quality_gated.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    ordered = [
        "STAGE smoke",
        "STAGE teacher-evaluation",
        "STAGE teacher-gate",
        "STAGE collection",
        "STAGE collection-audit",
        "STAGE dataset",
        "STAGE training",
        "STAGE results",
    ]
    positions = [output.index(marker) for marker in ordered]
    assert positions == sorted(positions)
    assert "--games 400" in output
    assert "--minimum-win-rate 0.58" in output
    assert "--simulations 128" in output
    assert "--particles 3" in output
    assert "--max-depth 10" in output
    assert "--resume" in output
    assert "scripts/promote" not in output.lower()
    assert "No checkpoint was promoted automatically." in output
