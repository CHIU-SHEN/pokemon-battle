"""Smoke-test bad-case conversion and its game-level split invariant."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bad_case_convert_") as tmp:
        output = Path(tmp) / "decisions.jsonl"
        summary = Path(tmp) / "summary.json"
        subprocess.run(
            [sys.executable, "scripts/convert_bad_cases.py", "--max-files", "3", "--with-v0", "--output", str(output), "--summary", str(summary)],
            cwd=ROOT,
            check=True,
        )
        report = json.loads(summary.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
        assert report["input_files"] == 3
        assert report["converted_samples"] == len(rows) > 0
        assert report["error_count"] == 0
        assert not report["game_overlap"]
        assert all(
            (row["teacher"]["v0_action"] is not None) == (row["current_player"] == 0)
            for row in rows
        )
        assert all(row["teacher"]["v1_search"] is None for row in rows)
        games = {}
        for row in rows:
            games.setdefault(row["game_id"], row["split"])
            assert games[row["game_id"]] == row["split"]
    print("OK: bad-case observed-decision conversion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
