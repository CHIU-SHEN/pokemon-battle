#!/usr/bin/env python3
"""Build a self-contained NumPy SL-1 GRU package with V0 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from build_sl0_submission import ROOT, SUBMISSION, portable_load


MAIN = '''from __future__ import annotations
import os
import sys

for _path in (os.getcwd(), "/kaggle_simulations/agent"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action

LEDGER = GameLedger()
POLICY = None
LAST_ACTION_SOURCE = "init"


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
        from gru_model_runtime import GRUPolicy
        POLICY = GRUPolicy(read_deck_csv(), window_length=16)
    return POLICY


def action_source():
    return LAST_ACTION_SOURCE


def agent(obs_dict):
    global LEDGER, LAST_ACTION_SOURCE
    if obs_dict is None or obs_dict.get("select") is None:
        if POLICY is not None:
            POLICY.reset()
        LEDGER = GameLedger()
        LAST_ACTION_SOURCE = "deck"
        return read_deck_csv()
    try:
        obs: Observation = to_observation_class(obs_dict)
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        try:
            action = get_policy().choose(parsed)
            if is_legal_action(obs.select, action):
                LAST_ACTION_SOURCE = "gru"
                return action
        except Exception:
            pass
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            LAST_ACTION_SOURCE = "rules_fallback"
            return action
        LAST_ACTION_SOURCE = "safe_fallback"
        return safe_action(obs.select, parsed, prefer_empty=False)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            LAST_ACTION_SOURCE = "exception_fallback"
            return safe_action(obs.select, prefer_empty=False) if obs.select is not None else read_deck_csv()
        except Exception:
            LAST_ACTION_SOURCE = "fatal_fallback"
            return []
'''


def build(checkpoint: Path, out_dir: Path, out_zip: Path) -> tuple[Path, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(SUBMISSION / "cg", out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(SUBMISSION / "agent", out_dir / "agent", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(SUBMISSION / "deck.csv", out_dir / "deck.csv")
    shutil.copy2(SUBMISSION / "model_runtime.py", out_dir / "gru_model_runtime.py")
    shutil.copy2(ROOT / "data/card_tags.json", out_dir / "card_tags.json")
    (out_dir / "main.py").write_text(MAIN, encoding="utf-8")
    checkpoint_doc = portable_load(checkpoint)
    if checkpoint_doc.get("schema_version") != "sl1_gru_checkpoint_v1":
        raise ValueError("checkpoint is not an SL-1 GRU checkpoint")
    arrays = {key: value.detach().cpu().numpy() for key, value in checkpoint_doc["model_state"].items()}
    np.savez_compressed(out_dir / "sl1_gru_best.npz", **arrays)
    metadata = {
        "schema_version": "sl1_gru_numpy_submission_v1",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_epoch": checkpoint_doc.get("epoch"),
        "dataset_sha256": checkpoint_doc.get("dataset_sha256"),
        "sequence_manifest_sha256": checkpoint_doc.get("sequence_manifest_sha256"),
        "model_config": checkpoint_doc.get("model_config"),
        "window_length": 16,
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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/sl1_gru_seed20260721/best.pt")
    parser.add_argument("--name", default="sl1_gru_seed20260721_stage1")
    args = parser.parse_args()
    out_dir = ROOT / "final_submissions" / args.name
    out_zip = ROOT / "final_submissions" / f"{args.name}.zip"
    print(build(args.checkpoint, out_dir, out_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
