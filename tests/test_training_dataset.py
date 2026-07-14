"""Build and validate a small formal training dataset."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="training_dataset_") as tmp:
        tmp_path = Path(tmp)
        bad = tmp_path / "bad.jsonl"
        kaggle = tmp_path / "kaggle.jsonl"
        v1 = tmp_path / "v1.jsonl"
        output = tmp_path / "training.jsonl"
        manifest = tmp_path / "manifest.json"

        v1_row = json.loads(next((ROOT / "data/reanalysis/v1_labels.jsonl").open(encoding="utf-8")))
        v1_sample_id = v1_row["sample_id"]
        bad_rows = []
        with (ROOT / "data/processed/bad_case_decisions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["sample_id"] == v1_sample_id:
                    bad_rows.append(row)
                    break
        kaggle_row = json.loads(next((ROOT / "data/processed/kaggle_decisions.jsonl").open(encoding="utf-8")))
        bad.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bad_rows), encoding="utf-8")
        kaggle.write_text(json.dumps(kaggle_row, ensure_ascii=False) + "\n", encoding="utf-8")
        v1.write_text(json.dumps(v1_row, ensure_ascii=False) + "\n", encoding="utf-8")

        subprocess.run(
            [sys.executable, "scripts/build_training_dataset.py", "--bad-cases", str(bad), "--kaggle", str(kaggle), "--v1", str(v1), "--output", str(output), "--manifest", str(manifest)],
            cwd=ROOT,
            check=True,
        )
        report = json.loads(manifest.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
        assert report["ok"]
        assert report["samples"] == len(rows) == 2
        assert report["policy_sources"]["v1_search"] == 1
        assert report["policy_sources"]["kaggle_agent"] == 1
        assert all(len(row["supervision"]["soft_policy"]) == row["select"]["option_count"] for row in rows)
    print("OK: formal training dataset merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
