#!/usr/bin/env python3
"""Freeze final submission candidates and closeout artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_flat_submission import build_flat_submission  # noqa: E402

FINAL_DIR = PROJECT_ROOT / "final_submissions"
REPORT_DIR = PROJECT_ROOT / "reports"

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    m5 = load_json(PROJECT_ROOT / "experiments" / "20260706_m5_elite001_v2_league_500" / "league_report.json")
    return {
        "version": "2026-07-14-clean-baseline-freeze",
        "primary_submission": "final_submissions/submission_flat_safe_v0.zip",
        "primary_reason": (
            "Flat single-file V0 is the safest Kaggle candidate because it is raw-exec "
            "compatible: no __file__, no agent package imports, no training dependencies."
        ),
        "frozen_paths": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in paths.items()},
        "promotion_decisions": {
            "flat_v0_rules": "promoted_primary",
            "multi_module_v0_rules": "source_only_not_packaged",
            "v1_search": "source_only_not_packaged",
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
        },
        "submission_package_rules": [
            "Primary Kaggle package is flat: main.py, deck.csv, and cg/ only.",
            "main.py must pass raw exec without __file__ in globals.",
            "No agent/ package imports in the primary Kaggle package.",
            "No training scripts, local data dependencies, pycache, pyc, or .DS_Store files.",
            "Multi-module V0 and Search V1 remain source-level research components and are not frozen as duplicate packages.",
        ],
    }


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    flat_dir, flat_zip = build_flat_submission()
    report = build_report(
        {
            "flat_safe_v0_dir": flat_dir,
            "flat_safe_v0_zip": flat_zip,
        }
    )
    report_path = REPORT_DIR / "final_freeze_report.json"
    write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "primary": report["primary_submission"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
