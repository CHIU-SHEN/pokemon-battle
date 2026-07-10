#!/usr/bin/env python3
"""Build a Kaggle raw-exec compatible flat single-file submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
OUT_DIR = PROJECT_ROOT / "final_submissions" / "submission_flat_safe_v0"
OUT_ZIP = PROJECT_ROOT / "final_submissions" / "submission_flat_safe_v0.zip"


HEADER = '''from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

for _path in (os.getcwd(), "/kaggle_simulations/agent"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import AreaType, Observation, OptionType, SelectContext, to_observation_class

'''


FOOTER = '''

LEDGER = GameLedger()


def read_deck_csv():
    for file_path in (
        "deck.csv",
        os.path.join(os.getcwd(), "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    return [
        1158,
        721,
        721,
        722,
        722,
        722,
        722,
        723,
        723,
        723,
        723,
        1145,
        1145,
        1145,
        1145,
        1205,
        1205,
        1227,
        1227,
        1227,
        1227,
        1235,
        1235,
        1235,
        1235,
    ] + [BASIC_WATER_ENERGY] * 35


def agent(obs_dict):
    if obs_dict is None:
        return read_deck_csv()
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
        return safe_action(obs.select, parsed, prefer_empty=False)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return read_deck_csv()
            return safe_action(obs.select, prefer_empty=False)
        except Exception:
            return []
'''


def strip_docstring(text: str) -> str:
    return re.sub(r"\A\s*(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')\s*", "", text)


def strip_imports(text: str, *, remove_relative_block: bool = False) -> str:
    text = strip_docstring(text)
    lines = text.splitlines()
    out: list[str] = []
    skip_block = False
    for line in lines:
        stripped = line.strip()
        if skip_block:
            if stripped == ")":
                skip_block = False
            continue
        if not stripped:
            out.append(line)
            continue
        if stripped.startswith("from __future__ import"):
            continue
        if stripped.startswith("from dataclasses import"):
            continue
        if stripped.startswith("from typing import"):
            continue
        if stripped.startswith("from cg.api import"):
            continue
        if stripped.startswith("from ."):
            if remove_relative_block and stripped.endswith("("):
                skip_block = True
            continue
        if stripped.startswith("import ") and stripped in {"import os", "import sys"}:
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def build_main() -> str:
    parts = [HEADER]
    for rel in [
        "agent/deck_profile_abomasnow.py",
        "agent/parser.py",
        "agent/fallback.py",
        "agent/rules.py",
    ]:
        source = (SUBMISSION_DIR / rel).read_text(encoding="utf-8")
        parts.append(f"\n# ---- inlined from {rel} ----\n")
        parts.append(strip_imports(source, remove_relative_block=True))
    parts.append(FOOTER)
    return "\n".join(parts)


def should_skip(path: Path) -> bool:
    return path.name == ".DS_Store" or "__pycache__" in path.parts or path.suffix == ".pyc"


def zip_dir(src: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(src.rglob("*")):
            if item.is_file() and not should_skip(item.relative_to(src)):
                zf.write(item, item.relative_to(src))


def build_flat_submission(
    out_dir: Path = OUT_DIR,
    out_zip: Path = OUT_ZIP,
    deck_path: Path | None = None,
) -> tuple[Path, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SUBMISSION_DIR / "cg", out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    shutil.copy2(deck_path or (SUBMISSION_DIR / "deck.csv"), out_dir / "deck.csv")
    (out_dir / "main.py").write_text(build_main(), encoding="utf-8")
    zip_dir(out_dir, out_zip)
    return out_dir, out_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=SUBMISSION_DIR / "deck.csv", help="deck csv to package")
    parser.add_argument("--name", default="safe_v0", help="suffix for final_submissions/submission_flat_<name>.zip")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-zip", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir or (PROJECT_ROOT / "final_submissions" / f"submission_flat_{args.name}")
    out_zip = args.out_zip or (PROJECT_ROOT / "final_submissions" / f"submission_flat_{args.name}.zip")
    out_dir, out_zip = build_flat_submission(out_dir=out_dir, out_zip=out_zip, deck_path=args.deck)
    print({"flat_dir": str(out_dir), "flat_zip": str(out_zip)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
