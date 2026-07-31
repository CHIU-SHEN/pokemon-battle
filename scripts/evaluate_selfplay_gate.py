#!/usr/bin/env python3
"""Evaluate an aggregated candidate-vs-best self-play Arena result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.selfplay_gate import gate_decision  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wins", type=int, required=True)
    parser.add_argument("--losses", type=int, required=True)
    parser.add_argument("--draws", type=int, default=0)
    parser.add_argument("--games-cap", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decision = gate_decision(
        args.wins,
        args.losses,
        args.draws,
        games_cap=args.games_cap,
    )
    payload = decision.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return {"promote_ready": 0, "reject": 2, "continue": 3}[decision.status]


if __name__ == "__main__":
    raise SystemExit(main())
