#!/usr/bin/env python3
"""Raw-exec tests for the flat Kaggle submission package."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def raw_exec_agent(main_path: Path):
    env = {"__builtins__": __builtins__}
    code = main_path.read_text(encoding="utf-8")
    exec(code, env)
    return env["agent"]


def main() -> int:
    subprocess.run([sys.executable, "scripts/build_flat_submission.py"], cwd=PROJECT_ROOT, check=True)
    flat_dir = PROJECT_ROOT / "final_submissions" / "submission_flat_safe_v0"
    main_path = flat_dir / "main.py"
    assert main_path.exists()
    assert "__file__" not in main_path.read_text(encoding="utf-8")
    assert "from agent." not in main_path.read_text(encoding="utf-8")

    old_cwd = Path.cwd()
    try:
        # Simulate Kaggle's raw exec style without __file__ in globals.
        import os

        os.chdir(flat_dir)
        agent = raw_exec_agent(main_path)
        deck = agent(None)
        assert isinstance(deck, list)
        assert len(deck) == 60
        assert all(isinstance(card_id, int) for card_id in deck)

        fixtures = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "observations.json").read_text(encoding="utf-8"))
        for obs in fixtures[:20]:
            action = agent(obs)
            select = obs["select"]
            assert isinstance(action, list)
            assert select["minCount"] <= len(action) <= select["maxCount"]
            assert len(set(action)) == len(action)
            assert all(isinstance(idx, int) and 0 <= idx < len(select["option"]) for idx in action)
    finally:
        import os

        os.chdir(old_cwd)

    # Validate raw exec after unpacking the zip into a fresh directory too.
    with tempfile.TemporaryDirectory(prefix="flat_submission_") as tmp:
        import zipfile

        zip_path = PROJECT_ROOT / "final_submissions" / "submission_flat_safe_v0.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "deck.csv" in names
            assert not any(name.startswith("agent/") for name in names)
            assert not any("__pycache__" in name or name.endswith(".pyc") or ".DS_Store" in name for name in names)
            zf.extractall(tmp)
        import os

        old = Path.cwd()
        try:
            os.chdir(tmp)
            agent = raw_exec_agent(Path(tmp) / "main.py")
            assert len(agent(None)) == 60
        finally:
            os.chdir(old)

    print("OK: flat raw-exec submission package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

