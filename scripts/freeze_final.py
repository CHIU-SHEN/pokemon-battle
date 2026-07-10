#!/usr/bin/env python3
"""Freeze final submission candidates and closeout artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_flat_submission import build_flat_submission  # noqa: E402

SUBMISSION_DIR = PROJECT_ROOT / "submission"
FINAL_DIR = PROJECT_ROOT / "final_submissions"
REPORT_DIR = PROJECT_ROOT / "reports"


SAFE_MAIN = '''import os

from cg.api import Observation, to_observation_class
from agent.fallback import is_legal_action, safe_action
from agent.parser import GameLedger, parse_observation
from agent.rules import choose_action


LEDGER = GameLedger()


def read_deck_csv():
    for file_path in ("deck.csv", os.path.join(os.getcwd(), "deck.csv"), "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return [int(line.strip()) for line in file if line.strip()]
    raise FileNotFoundError("deck.csv not found")


def agent(obs_dict):
    if obs_dict is None:
        return read_deck_csv()
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        parsed = parse_observation(obs_dict)
        LEDGER.update(parsed)
        action = choose_action(parsed)
        if is_legal_action(obs.select, action):
            return action
    except Exception:
        pass
    return safe_action(obs.select, prefer_empty=False)
'''


def should_skip(path: Path) -> bool:
    return path.name == ".DS_Store" or "__pycache__" in path.parts or path.suffix == ".pyc"


def copy_tree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if should_skip(rel):
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def freeze_safe_v0() -> Path:
    dst = FINAL_DIR / "submission_safe_v0"
    copy_tree_clean(SUBMISSION_DIR, dst)
    write_text(dst / "main.py", SAFE_MAIN)
    for rel in [
        "agent/action_gen.py",
        "agent/belief.py",
        "agent/search.py",
        "agent/value.py",
        "local_eval.py",
        "test_sim.py",
    ]:
        unused = dst / rel
        if unused.exists():
            unused.unlink()
    return dst


def freeze_search_v1() -> Path:
    dst = FINAL_DIR / "submission_search_v1_experimental"
    copy_tree_clean(SUBMISSION_DIR, dst)
    main_path = dst / "main.py"
    text = main_path.read_text(encoding="utf-8")
    text = text.replace('os.environ.get("PTCG_ENABLE_SEARCH", "0")', 'os.environ.get("PTCG_ENABLE_SEARCH", "1")')
    write_text(main_path, text)
    for rel in ["local_eval.py", "test_sim.py"]:
        unused = dst / rel
        if unused.exists():
            unused.unlink()
    return dst


def freeze_v2_archive() -> Path:
    dst = FINAL_DIR / "v2_distill_experimental"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for src, rel in [
        (PROJECT_ROOT / "models" / "v2_policy_linear.json", "models/v2_policy_linear.json"),
        (PROJECT_ROOT / "data" / "distill" / "schema.json", "data/distill/schema.json"),
        (PROJECT_ROOT / "data" / "distill" / "train_metrics.json", "data/distill/train_metrics.json"),
        (PROJECT_ROOT / "data" / "reanalysis_queue.json", "data/reanalysis_queue.json"),
        (PROJECT_ROOT / "M6_小模型蒸馏说明.md", "M6_小模型蒸馏说明.md"),
    ]:
        if src.exists():
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    write_text(
        dst / "README.md",
        "# V2 distillation experimental archive\n\n"
        "This archive is not a Kaggle submission. The model is marked "
        "`experimental_not_promoted` and is kept only for M6 reproducibility.\n",
    )
    return dst


def zip_dir(src: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(src.rglob("*")):
            if item.is_file() and not should_skip(item.relative_to(src)):
                zf.write(item, item.relative_to(src))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_match(path: str) -> dict:
    obj = load_json(PROJECT_ROOT / path)
    return {
        "path": path,
        "games": obj.get("games"),
        "agent0_wins": obj.get("agent0_wins"),
        "agent1_wins": obj.get("agent1_wins"),
        "draws": obj.get("draws"),
        "agent0_win_rate_all": obj.get("agent0_win_rate_all"),
        "p95_agent_time_sec_per_decision": obj.get("p95_agent_time_sec_per_decision"),
        "exceptions": obj.get("exceptions"),
        "illegal_actions": obj.get("illegal_actions"),
    }


def build_report(paths: dict[str, Path]) -> dict:
    m3_stability = compact_match("experiments/20260706_m3_search_5000_stability/summary.json")
    distill = load_json(PROJECT_ROOT / "data" / "distill" / "train_metrics.json")
    m5 = load_json(PROJECT_ROOT / "experiments" / "20260706_m5_elite001_v2_league_500" / "league_report.json")
    return {
        "version": "2026-07-07-final-freeze",
        "primary_submission": "final_submissions/submission_flat_safe_v0.zip",
        "primary_reason": (
            "Flat single-file V0 is the safest Kaggle candidate because it is raw-exec "
            "compatible: no __file__, no agent package imports, no training dependencies."
        ),
        "frozen_paths": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in paths.items()},
        "promotion_decisions": {
            "flat_v0_rules": "promoted_primary",
            "multi_module_v0_rules": "local_backup_only",
            "v1_search": "experimental_backup_only",
            "v2_distill": "not_kaggle_submission",
            "m5_deck_elite": "not_promoted",
        },
        "key_evidence": {
            "final_flat_safe_v0_vs_random_200": compact_match("experiments/final_flat_safe_v0_vs_random_200/summary.json"),
            "final_flat_safe_v0_selfplay_50": compact_match("experiments/final_flat_safe_v0_selfplay_50/summary.json"),
            "final_safe_v0_vs_random_200": compact_match("experiments/final_safe_v0_vs_random_200/summary.json"),
            "final_search_v1_vs_random_100": compact_match("experiments/final_search_v1_vs_random_100/summary.json"),
            "final_audit_safe_v0_vs_random_200": compact_match("experiments/final_audit_safe_v0_vs_random_200/summary.json"),
            "final_audit_search_v1_vs_random_100": compact_match("experiments/final_audit_search_v1_vs_random_100/summary.json"),
            "m3_search_5000_stability": m3_stability,
            "m5_elite001_v2_league_500_baselines": {
                name: {
                    "win_rate": item.get("agent0_win_rate_all"),
                    "record": [item.get("agent0_wins"), item.get("agent1_wins"), item.get("draws")],
                    "exceptions": item.get("exceptions"),
                    "illegal_actions": item.get("illegal_actions"),
                }
                for name, item in (m5.get("matrix") or {}).items()
            },
            "m6_distill_smoke": {
                "train_samples": (distill.get("split") or {}).get("train_samples"),
                "valid_samples": (distill.get("split") or {}).get("valid_samples"),
                "valid_policy_top1": (distill.get("valid") or {}).get("policy_top1"),
                "valid_policy_top3": (distill.get("valid") or {}).get("policy_top3"),
                "valid_value_corr": (distill.get("valid") or {}).get("value_target_corr"),
                "latency_ms_per_select": distill.get("latency_ms_per_select"),
            },
        },
        "submission_package_rules": [
            "Primary Kaggle package is flat: main.py, deck.csv, and cg/ only.",
            "main.py must pass raw exec without __file__ in globals.",
            "No agent/ package imports in the primary Kaggle package.",
            "No training scripts, local data dependencies, pycache, pyc, or .DS_Store files.",
            "Multi-module V0 and Search V1 are local/experimental backups only.",
            "V2 distillation archive is reproducibility-only and not meant for upload.",
        ],
    }


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    flat_dir, flat_zip = build_flat_submission()
    safe = freeze_safe_v0()
    search = freeze_search_v1()
    v2 = freeze_v2_archive()
    zip_dir(safe, FINAL_DIR / "submission_safe_v0.zip")
    zip_dir(search, FINAL_DIR / "submission_search_v1_experimental.zip")
    zip_dir(v2, FINAL_DIR / "v2_distill_experimental.zip")
    report = build_report(
        {
            "flat_safe_v0_dir": flat_dir,
            "flat_safe_v0_zip": flat_zip,
            "safe_v0_dir": safe,
            "search_v1_dir": search,
            "v2_distill_dir": v2,
            "safe_v0_zip": FINAL_DIR / "submission_safe_v0.zip",
            "search_v1_zip": FINAL_DIR / "submission_search_v1_experimental.zip",
            "v2_distill_zip": FINAL_DIR / "v2_distill_experimental.zip",
        }
    )
    report_path = REPORT_DIR / "final_freeze_report.json"
    write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "primary": report["primary_submission"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
