"""Audit locally available training data and materialize directly derivable indexes."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def deck_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [int(row[0]) for row in csv.reader(f) if row]


def main() -> None:
    cards_doc = load(ROOT / "data/cards.json")
    cards = cards_doc["cards"]
    tags = load(ROOT / "data/card_tags.json")["cards"]
    target = deck_ids(ROOT / "submission/deck.csv")
    counts = Counter(target)
    missing_cards = sorted(i for i in counts if str(i) not in cards)
    missing_tags = sorted(i for i in counts if str(i) not in tags)

    target_rows = []
    for card_id, count in sorted(counts.items()):
        card = cards.get(str(card_id), {})
        target_rows.append({
            "card_id": card_id,
            "count": count,
            "name_en": card.get("name_en"),
            "name_jp": card.get("name_jp"),
            "stage_type": card.get("stage_type"),
            "hp": card.get("hp"),
            "tags": tags.get(str(card_id), {}).get("tags", []),
        })

    bad_files = sorted((ROOT / "logs/bad_cases").rglob("*.json"))
    reasons, matchups = Counter(), Counter()
    total_steps = illegal = parse_errors = complete_trace = 0
    bad_index = []
    for path in bad_files:
        try:
            doc = load(path)
            rec = doc.get("record", {})
            trace = rec.get("trace") or []
            rs = doc.get("reasons") or ["unlabelled"]
            reasons.update(rs)
            matchup = doc.get("matchup", {})
            matchup_name = f'{matchup.get("agent0", "?")}_vs_{matchup.get("agent1", "?")}'
            matchups[matchup_name] += 1
            steps = int(rec.get("steps", len(trace)))
            total_steps += len(trace)
            illegal += sum(rec.get("illegal_actions") or [])
            complete_trace += int(bool(trace) and len(trace) == steps)
            bad_index.append({"case_id": doc.get("case_id"), "path": path.relative_to(ROOT).as_posix(),
                              "matchup": matchup_name, "result": rec.get("result"),
                              "steps": steps, "trace_steps": len(trace), "reasons": rs})
        except (OSError, ValueError, TypeError):
            parse_errors += 1

    summaries = list((ROOT / "experiments").rglob("summary.json"))
    game_files = list((ROOT / "experiments").rglob("games.json"))
    game_records = 0
    for path in game_files:
        try:
            doc = load(path)
            game_records += len(doc if isinstance(doc, list) else doc.get("games", []))
        except (OSError, ValueError, TypeError):
            pass

    observed_path = ROOT / "data/processed/bad_case_decisions.jsonl"
    observed_summary_path = ROOT / "data/processed/bad_case_conversion_summary.json"
    kaggle_summary_path = ROOT / "data/processed/kaggle_conversion_summary.json"
    v1_summary_path = ROOT / "data/reanalysis/v1_labels_summary.json"
    observed_summary = {}
    observed_samples = 0
    observed_games: set[str] = set()
    observed_splits = Counter()
    if observed_path.exists():
        with observed_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                observed_samples += 1
                observed_games.add(row["game_id"])
                observed_splits[row.get("split")] += 1
    if observed_summary_path.exists():
        observed_summary = load(observed_summary_path)
    kaggle_summary = load(kaggle_summary_path) if kaggle_summary_path.exists() else {}
    v1_summary = load(v1_summary_path) if v1_summary_path.exists() else {}
    training_manifest_path = ROOT / "data/training/training_manifest_v1.json"
    training_manifest = load(training_manifest_path) if training_manifest_path.exists() else {}
    deck_selection_path = ROOT / "data/high_score_decks/selection_manifest.json"
    deck_selection = load(deck_selection_path) if deck_selection_path.exists() else {}
    stage_b_path = ROOT / "experiments/high_score_deck_selection/stage_b/stage_b_report.json"
    stage_b = load(stage_b_path) if stage_b_path.exists() else {}
    deck_hash = hashlib.sha256("\n".join(map(str, sorted(target))).encode()).hexdigest()
    result = {
        "audit_version": "local_data_audit_v1",
        "target_deck": {"cards": len(target), "unique_cards": len(counts), "sha256_sorted_ids": deck_hash,
                        "role": "incumbent_submission_pending_top10_selection",
                        "missing_card_records": missing_cards, "missing_tag_records": missing_tags,
                        "valid_60_cards": len(target) == 60, "cards": target_rows},
        "card_database": {"card_count": len(cards), "tagged_card_count": len(tags),
                          "tag_coverage": round(len(tags) / len(cards), 6) if cards else 0},
        "bad_cases": {"files": len(bad_files), "parse_errors": parse_errors,
                      "complete_trace_files": complete_trace, "trace_decisions": total_steps,
                      "illegal_actions": illegal, "reasons": dict(reasons), "matchups": dict(matchups)},
        "observed_bad_case_decisions": {
            "schema_version": "observed_decision_v1",
            "samples": observed_samples,
            "games": len(observed_games),
            "split_samples": dict(observed_splits),
            "teacher_status": observed_summary.get("teacher_status", "not_generated"),
            "v0_teacher_samples": observed_summary.get("v0_teacher_samples", 0),
        },
        "observed_kaggle_decisions": {
            "schema_version": kaggle_summary.get("schema_version"),
            "samples": kaggle_summary.get("converted_samples", 0),
            "games": kaggle_summary.get("converted_games", 0),
            "split_games": kaggle_summary.get("split_games", {}),
            "teacher_status": kaggle_summary.get("teacher_status"),
            "error_count": kaggle_summary.get("error_count"),
        },
        "v1_search_labels": {
            "schema_version": v1_summary.get("schema_version"),
            "requested": v1_summary.get("requested", 0),
            "labelled": v1_summary.get("labelled", 0),
            "failed": v1_summary.get("failed", 0),
            "search_used": v1_summary.get("search_used", 0),
            "v1_changed_v0": v1_summary.get("v1_changed_v0", 0),
            "budget": v1_summary.get("budget", {}),
        },
        "formal_training_dataset": {
            "schema_version": training_manifest.get("schema_version"),
            "samples": training_manifest.get("samples", 0),
            "games": training_manifest.get("games", 0),
            "splits": training_manifest.get("splits", {}),
            "policy_sources": training_manifest.get("policy_sources", {}),
            "sha256": training_manifest.get("sha256"),
            "unique_sample_ids": training_manifest.get("unique_sample_ids", 0),
            "cross_split_games": training_manifest.get("cross_split_games", []),
            "unused_v1_labels": training_manifest.get("unused_v1_labels"),
            "ok": training_manifest.get("ok", False),
        },
        "high_score_deck_selection": {
            "candidate_count": deck_selection.get("candidate_count", 0),
            "static_valid_count": deck_selection.get("static_valid_count", 0),
            "mapping_complete_count": deck_selection.get("mapping_complete_count", 0),
            "replay_prior_leader": deck_selection.get("replay_prior_leader"),
            "promotion_decision": deck_selection.get("promotion_decision"),
            "stage_b_hard_gate_pass_count": stage_b.get("hard_gate_pass_count", 0),
            "common_policy_priority": stage_b.get("common_policy_priority", stage_b.get("stage_c_candidates", [])),
            "adapter_training_candidates": stage_b.get("adapter_training_candidates", []),
            "ok": deck_selection.get("ok", False),
        },
        "experiments": {"summary_files": len(summaries), "game_files": len(game_files),
                        "game_records": game_records},
        "deck_library": {"elite_decks": len(list((ROOT / "data/deck_elites").glob("*.csv")))},
    }
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "target_deck_profile.json").write_text(json.dumps(result["target_deck"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "bad_case_index.json").write_text(json.dumps({"items": bad_index}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
