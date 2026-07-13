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

    smoke = (ROOT / "data/distill/smoke_samples.jsonl").read_text(encoding="utf-8").splitlines()
    smoke_docs = [json.loads(x) for x in smoke if x.strip()]
    deck_hash = hashlib.sha256("\n".join(map(str, sorted(target))).encode()).hexdigest()
    result = {
        "audit_version": "local_data_audit_v1",
        "target_deck": {"cards": len(target), "unique_cards": len(counts), "sha256_sorted_ids": deck_hash,
                        "missing_card_records": missing_cards, "missing_tag_records": missing_tags,
                        "valid_60_cards": len(target) == 60, "cards": target_rows},
        "card_database": {"card_count": len(cards), "tagged_card_count": len(tags),
                          "tag_coverage": round(len(tags) / len(cards), 6) if cards else 0},
        "bad_cases": {"files": len(bad_files), "parse_errors": parse_errors,
                      "complete_trace_files": complete_trace, "trace_decisions": total_steps,
                      "illegal_actions": illegal, "reasons": dict(reasons), "matchups": dict(matchups)},
        "distill": {"samples": len(smoke_docs), "games": len({x["game_id"] for x in smoke_docs})},
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
