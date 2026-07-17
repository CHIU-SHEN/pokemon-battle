#!/usr/bin/env python3
"""Build a self-contained Kaggle package using NumPy SL-0 inference with V0 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


MAIN = '''from __future__ import annotations
import os
import sys
from typing import Any

for _path in (os.getcwd(), "/kaggle_simulations/agent"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action

LEDGER = GameLedger()
POLICY = None


def read_deck_csv():
    for root in (os.getcwd(), "/kaggle_simulations/agent"):
        path = os.path.join(root, "deck.csv")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as stream:
                deck = [int(line.strip()) for line in stream if line.strip()]
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("deck.csv")


def get_policy():
    global POLICY
    if POLICY is None:
        from model_runtime import SL0Policy
        POLICY = SL0Policy(read_deck_csv())
    return POLICY


def agent(obs_dict):
    if obs_dict is None:
        return read_deck_csv()
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        try:
            action = get_policy().choose(parsed)
            if is_legal_action(obs.select, action):
                return action
        except Exception:
            pass
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
        return safe_action(obs.select, parsed, prefer_empty=False)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            return safe_action(obs.select, prefer_empty=False) if obs.select is not None else read_deck_csv()
        except Exception:
            return []
'''


def portable_load(path: Path):
    import pathlib
    original = pathlib.PosixPath
    if __import__("sys").platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original


def build(checkpoint: Path, out_dir: Path, out_zip: Path) -> tuple[Path, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(SUBMISSION / "cg", out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(SUBMISSION / "agent", out_dir / "agent", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(SUBMISSION / "deck.csv", out_dir / "deck.csv")
    shutil.copy2(SUBMISSION / "model_runtime.py", out_dir / "model_runtime.py")
    shutil.copy2(ROOT / "data/card_tags.json", out_dir / "card_tags.json")
    (out_dir / "main.py").write_text(MAIN, encoding="utf-8")
    checkpoint_doc = portable_load(checkpoint)
    arrays = {key: value.detach().cpu().numpy() for key, value in checkpoint_doc["model_state"].items()}
    np.savez_compressed(out_dir / "sl0_shared_best.npz", **arrays)
    metadata = {
        "schema_version": "sl0_numpy_submission_v1",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_schema": checkpoint_doc.get("schema_version"),
        "checkpoint_epoch": checkpoint_doc.get("epoch"),
        "dataset_sha256": checkpoint_doc.get("dataset_sha256"),
        "model_config": checkpoint_doc.get("model_config"),
        "policy_gate": "mandatory single-choice with more than one option; V0 fallback otherwise",
    }
    (out_dir / "MODEL_INFO.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(out_dir.rglob("*")):
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                archive.write(item, item.relative_to(out_dir))
    return out_dir, out_zip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/sl0_shared_full/best.pt")
    parser.add_argument("--name", default="sl0_shared_stage1")
    args = parser.parse_args()
    out_dir = ROOT / "final_submissions" / args.name
    out_zip = ROOT / "final_submissions" / f"{args.name}.zip"
    print(build(args.checkpoint, out_dir, out_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
