#!/usr/bin/env python3
"""Validate Arena stages and write the reproducible Top2 selection report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRELIMINARY = ROOT / "reports" / "top2_arena_report.json"
PLAYOFF = ROOT / "reports" / "top4_playoff_report.json"
FINAL = ROOT / "reports" / "top2_final_report.json"
OUT_JSON = ROOT / "reports" / "top2_freeze_report.json"
OUT_MD = ROOT / "reports" / "top2_freeze_report.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage(name: str, report: dict[str, Any]) -> None:
    if not report.get("complete"):
        raise ValueError(f"{name} is incomplete")
    if report.get("failures"):
        raise ValueError(f"{name} contains failures")
    if any(not row.get("hard_gate_pass") for row in report["ranking"]):
        raise ValueError(f"{name} contains a candidate that failed hard gates")


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row["rank"],
        "candidate": row["candidate"],
        "composite_score": row["composite_score"],
        "internal_score": row["internal_score"],
        "internal_games": row["internal_games"],
        "internal_wilson_95": row["internal_wilson_95"],
        "external_score": row["external_score"],
        "external_games": row["external_games"],
        "worst_matchup_score": row["worst_matchup_score"],
        "max_p95_decision_seconds": row["max_p95_decision_seconds"],
        "exceptions": row["exceptions"],
        "illegal_actions": row["illegal_actions"],
        "hard_gate_pass": row["hard_gate_pass"],
    }


def role_record(role: str, candidate: str, stage_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deck = ROOT / "data" / "high_score_decks" / candidate / "deck.csv"
    adapter = ROOT / "artifacts" / "adapters_top10" / candidate / "best.pt"
    return {
        "role": role,
        "candidate": candidate,
        "deck": str(deck.relative_to(ROOT)),
        "deck_sha256": sha256(deck),
        "adapter_checkpoint": str(adapter.relative_to(ROOT)),
        "adapter_sha256": sha256(adapter),
        "stage_results": stage_rows,
    }


def main() -> int:
    preliminary = read_json(PRELIMINARY)
    playoff = read_json(PLAYOFF)
    final = read_json(FINAL)
    for name, report in (("preliminary", preliminary), ("playoff", playoff), ("final", final)):
        validate_stage(name, report)

    prelim_top2 = [row["candidate"] for row in preliminary["ranking"][:2]]
    playoff_top2 = [row["candidate"] for row in playoff["ranking"][:2]]
    final_top2 = [row["candidate"] for row in final["ranking"][:2]]
    if not (prelim_top2 == playoff_top2 == final_top2):
        raise ValueError(
            f"Top2 order is not stable across stages: {prelim_top2}, {playoff_top2}, {final_top2}"
        )

    stage_maps = {
        "preliminary": {row["candidate"]: compact_row(row) for row in preliminary["ranking"]},
        "playoff": {row["candidate"]: compact_row(row) for row in playoff["ranking"]},
        "final": {row["candidate"]: compact_row(row) for row in final["ranking"]},
    }
    roles = []
    for role, candidate in zip(("primary", "reserve"), final_top2, strict=True):
        roles.append(
            role_record(
                role,
                candidate,
                {stage: rows[candidate] for stage, rows in stage_maps.items()},
            )
        )

    second = preliminary["ranking"][1]
    third = preliminary["ranking"][2]
    second_third_separated = second["internal_wilson_95"][0] > third["internal_wilson_95"][1]
    report = {
        "schema_version": "top2_freeze_report_v1",
        "decision": "top2_selected_with_external_baseline_caveat",
        "selection_complete": True,
        "release_submission_unchanged": True,
        "total_games": sum(stage["games_completed"] for stage in (preliminary, playoff, final)),
        "stage_games": {
            "preliminary": preliminary["games_completed"],
            "top4_playoff": playoff["games_completed"],
            "top2_final": final["games_completed"],
        },
        "top2_order_stable_across_stages": True,
        "rank2_vs_rank3_preliminary_intervals_separated": second_third_separated,
        "roles": roles,
        "caveats": [
            "Sample baseline source is absent locally, so the external matrix used Random, Exploiter-FirstMin, V0-current, and V0-best.",
            "The bundled battle engine does not expose RNG seed control; seat swapping and large samples were used, but games are not strictly paired by engine seed.",
            "This report freezes the Top2 selection only and does not authorize training or replacement of submission/deck.csv.",
        ],
        "source_reports": {
            "preliminary": str(PRELIMINARY.relative_to(ROOT)),
            "top4_playoff": str(PLAYOFF.relative_to(ROOT)),
            "top2_final": str(FINAL.relative_to(ROOT)),
        },
        "source_report_sha256": {
            "preliminary": sha256(PRELIMINARY),
            "top4_playoff": sha256(PLAYOFF),
            "top2_final": sha256(FINAL),
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    primary, reserve = roles
    final_primary = primary["stage_results"]["final"]
    final_reserve = reserve["stage_results"]["final"]
    markdown = f"""# Top2 Arena 筛选结果

- 决策：已选出 Top2；仅冻结筛选结果，不启动训练，不覆盖 `submission/deck.csv`。
- 总对局数：{report['total_games']:,}（首轮 {preliminary['games_completed']:,} + 前四复赛 {playoff['games_completed']:,} + 前二决赛 {final['games_completed']:,}）。
- 全部阶段：0 失败、0 非法动作，Top2 顺序三阶段一致。

## 最终角色

1. Primary：`{primary['candidate']}`
2. Reserve：`{reserve['candidate']}`

决赛中 Primary 对 Reserve 的双座次合计胜分率为 {final_primary['internal_score']:.2%}，战绩为 {int(final_primary['internal_score'] * final_primary['internal_games'])}:{int(final_reserve['internal_score'] * final_reserve['internal_games'])}；95% Wilson 区间为 {final_primary['internal_wilson_95'][0]:.2%}–{final_primary['internal_wilson_95'][1]:.2%}。

## 重要边界

- 本地缺少 Sample baseline 源码；外部矩阵实际使用 Random、Exploiter-FirstMin、V0-current 和 V0-best。
- 对战引擎不暴露 RNG seed，已通过交换先后手和扩大样本降低偏差，但不属于严格成对 seed 实验。
- 后续训练地点与流程由项目负责人另行决定。
"""
    OUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "total_games": report["total_games"], "roles": final_top2}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
