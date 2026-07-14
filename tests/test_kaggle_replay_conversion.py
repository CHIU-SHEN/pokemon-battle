"""Smoke-test Kaggle replay action/observation alignment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="kaggle_convert_") as tmp:
        output = Path(tmp) / "samples.jsonl"
        summary = Path(tmp) / "summary.json"
        subprocess.run(
            [sys.executable, "scripts/convert_kaggle_replays.py", "--max-files", "3", "--output", str(output), "--summary", str(summary)],
            cwd=ROOT,
            check=True,
        )
        report = json.loads(summary.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
        assert report["converted_samples"] == len(rows) > 0
        assert report["converted_games"] == 3
        assert report["error_count"] == 0
        assert all(row["quality"]["action_aligned_from_next_record"] for row in rows)
        assert all(row["teacher"] == {"v0_action": None, "v1_search": None} for row in rows)
        assert all(len(row["legal_mask"]) == row["select"]["option_count"] for row in rows)
    print("OK: Kaggle replay conversion and delayed-action alignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
