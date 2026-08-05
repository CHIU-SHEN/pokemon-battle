#!/usr/bin/env python3
"""Gate a direct MCTS teacher evaluation before expensive collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.mcts_teacher_gate import teacher_gate_decision  # noqa: E402


def atomic_write_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-games", type=int, default=400)
    parser.add_argument("--minimum-win-rate", type=float, default=0.58)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    decision = teacher_gate_decision(
        report,
        minimum_games=args.minimum_games,
        minimum_win_rate=args.minimum_win_rate,
    )
    atomic_write_json(args.output, decision)
    print(json.dumps(decision, ensure_ascii=False))
    return 0 if decision["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
