"""Convert replayable bad cases into leakage-safe observed decision samples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
for path in (ROOT, SUBMISSION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent.parser import parse_observation  # noqa: E402
from agent.fallback import is_legal_action  # noqa: E402
from agent.rules import choose_action  # noqa: E402
from src.train.features import load_card_tags, sample_features  # noqa: E402
from src.train.observed_schema import (  # noqa: E402
    SCHEMA_VERSION,
    split_for_game,
    validate_sample,
    write_jsonl,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_history(observation: dict[str, Any]) -> list[dict[str, Any]]:
    logs = observation.get("logs") or []
    return [dict(item) for item in logs if isinstance(item, dict)]


def _result_targets(result: Any, player: int) -> tuple[str | None, float | None]:
    if result in (0, 1):
        winner = int(result)
        return ("win" if winner == player else "loss", 1.0 if winner == player else -1.0)
    if result in (-1, None):
        return None, None
    return "draw", 0.0


def convert_file(path: Path, card_tags: dict[int, set[str]], with_v0: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc = _load(path)
    record = doc.get("record") or {}
    # case_id is not globally unique: two matchup folders currently contain
    # different games named 20260706_152640_00029_agent0_loss. Include the
    # source folder so distinct games never collide or share a split key.
    relative = path.relative_to(ROOT)
    case_id = str(doc.get("case_id") or path.stem)
    game_id = f"{relative.parent.as_posix()}:{case_id}"
    split = split_for_game(game_id)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for ordinal, item in enumerate(record.get("trace") or []):
        observation = item.get("observation")
        action = item.get("action")
        if not isinstance(observation, dict) or not isinstance(action, list):
            errors.append({"game_id": game_id, "step": ordinal, "errors": ["missing observation or action"]})
            continue
        try:
            parsed = parse_observation(observation)
            if parsed.select is None:
                raise ValueError("observation has no select")
            player = int(item.get("player", parsed.current_player))
            matchup = doc.get("matchup") or {}
            is_target_submission = player == 0 and matchup.get("agent0") == "submission"
            v0_action = choose_action(parsed) if with_v0 and is_target_submission else None
            if v0_action is not None and not is_legal_action(parsed.select, v0_action):
                raise ValueError(f"V0 produced illegal action: {v0_action}")
            global_vec, option_vecs, rule_scores, options = sample_features(parsed, card_tags)
            game_result, value_target = _result_targets(record.get("result"), player)
            row = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"{game_id}:{int(item.get('step', ordinal)):04d}",
                "game_id": game_id,
                "source": {"type": "local_bad_case", "path": relative.as_posix(), "case_id": case_id},
                "split": split,
                "step": int(item.get("step", ordinal)),
                "turn": parsed.turn,
                "current_player": player,
                "select": {
                    "type": parsed.select.type,
                    "context": parsed.select.context,
                    "min_count": parsed.select.min_count,
                    "max_count": parsed.select.max_count,
                    "option_count": len(parsed.select.options),
                },
                "legal_mask": [True] * len(parsed.select.options),
                "observed_action": [int(i) for i in action],
                "teacher": {
                    "v0_action": [int(i) for i in v0_action] if v0_action is not None else None,
                    "v1_search": None,
                },
                "game_result": game_result,
                "value_target": value_target,
                "features": global_vec,
                "options": options,
                "option_features": option_vecs,
                "rule_scores_unverified": rule_scores,
                "public_history": _public_history(observation),
                "deck": {"player": None, "opponent": None, "note": "deck ids were not stored in this bad case"},
                "quality": {
                    "complete_game_trace": True,
                    "visible_observation_only": True,
                    "teacher_verified": v0_action is not None,
                    "forced_single_option": (
                        len(parsed.select.options) == 1
                        and parsed.select.min_count == 1
                        and parsed.select.max_count == 1
                    ),
                },
            }
            sample_errors = validate_sample(row)
            if sample_errors:
                errors.append({"game_id": game_id, "step": ordinal, "errors": sample_errors})
            else:
                rows.append(row)
        except Exception as exc:
            errors.append({"game_id": game_id, "step": ordinal, "errors": [f"{type(exc).__name__}: {exc}"]})
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "logs" / "bad_cases")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "bad_case_decisions.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "processed" / "bad_case_conversion_summary.json")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--with-v0", action="store_true", help="Generate and legality-check current V0 rule actions.")
    args = parser.parse_args()

    files = sorted(args.input.rglob("*.json"))
    if args.max_files:
        files = files[: args.max_files]
    tags = load_card_tags()
    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    for path in files:
        rows, errors = convert_file(path, tags, with_v0=args.with_v0)
        all_rows.extend(rows)
        all_errors.extend(errors)
    write_jsonl(args.output, all_rows)
    split_samples = Counter(row["split"] for row in all_rows)
    split_games: dict[str, set[str]] = {name: set() for name in ("train", "valid", "test")}
    for row in all_rows:
        split_games[row["split"]].add(row["game_id"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_files": len(files),
        "converted_samples": len(all_rows),
        "error_count": len(all_errors),
        "split_samples": dict(split_samples),
        "split_games": {key: len(value) for key, value in split_games.items()},
        "forced_single_option_samples": sum(
            bool(row["quality"]["forced_single_option"]) for row in all_rows
        ),
        "game_overlap": bool(
            split_games["train"] & split_games["valid"]
            or split_games["train"] & split_games["test"]
            or split_games["valid"] & split_games["test"]
        ),
        "teacher_status": "v0_rules_generated" if args.with_v0 else "not_generated",
        "v0_teacher_samples": sum(row["teacher"]["v0_action"] is not None for row in all_rows),
        "errors": all_errors[:100],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all_rows and not all_errors and not summary["game_overlap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
