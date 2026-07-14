"""Convert Kaggle simulation replays into aligned observed-decision samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
for path in (ROOT, SUBMISSION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent.parser import parse_observation  # noqa: E402
from src.train.features import load_card_tags, sample_features  # noqa: E402
from src.train.observed_schema import SCHEMA_VERSION, split_for_game, validate_sample  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_action(select: Any, action: Any) -> bool:
    if not isinstance(select, dict) or not isinstance(action, list):
        return False
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", minimum) or minimum)
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(i, int) and 0 <= i < len(options) for i in action)
    )


def _decks(replay: dict[str, Any]) -> list[list[int] | None]:
    try:
        actions = replay["steps"][0][0]["visualize"][0]["action"]
        if isinstance(actions, list) and len(actions) == 2:
            return [[int(x) for x in deck] if isinstance(deck, list) else None for deck in actions]
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return [None, None]


def _deck_info(deck: list[int] | None) -> dict[str, Any]:
    if deck is None:
        return {"cards": None, "sha256_sorted_ids": None}
    canonical = "\n".join(str(x) for x in sorted(deck))
    return {
        "cards": deck,
        "sha256_sorted_ids": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _result(rewards: Any, player: int) -> tuple[str | None, float | None]:
    if not isinstance(rewards, list) or player >= len(rewards) or rewards[player] is None:
        return None, None
    value = float(rewards[player])
    return ("win" if value > 0 else "loss" if value < 0 else "draw", max(-1.0, min(1.0, value)))


def convert_replay(path: Path, tags: dict[int, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    replay = _load(path)
    numeric_id = path.name.split("-")[1]
    game_id = f"kaggle_episode_{numeric_id}"
    split = split_for_game(game_id)
    teams = (replay.get("info") or {}).get("TeamNames") or [None, None]
    decks = _decks(replay)
    pending: list[dict[str, Any] | None] = [None, None]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for replay_step, records in enumerate(replay.get("steps") or []):
        if not isinstance(records, list):
            continue
        for player, record in enumerate(records[:2]):
            if not isinstance(record, dict):
                continue
            action = record.get("action")
            previous = pending[player]
            if previous is not None and _valid_action(previous.get("select"), action):
                try:
                    parsed = parse_observation(previous)
                    if parsed.select is None:
                        raise ValueError("pending observation has no select")
                    global_vec, option_vecs, rule_scores, options = sample_features(parsed, tags)
                    game_result, value_target = _result(replay.get("rewards"), player)
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": f"{game_id}:p{player}:s{len(rows):05d}",
                        "game_id": game_id,
                        "source": {
                            "type": "kaggle_official_replay",
                            "path": path.relative_to(ROOT).as_posix(),
                            "episode_id": numeric_id,
                            "team_name": teams[player] if player < len(teams) else None,
                        },
                        "split": split,
                        "step": int(previous.get("step", replay_step - 1) or replay_step - 1),
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
                        "teacher": {"v0_action": None, "v1_search": None},
                        "game_result": game_result,
                        "value_target": value_target,
                        "features": global_vec,
                        "options": options,
                        "option_features": option_vecs,
                        "rule_scores_unverified": rule_scores,
                        "public_history": [dict(x) for x in previous.get("logs") or [] if isinstance(x, dict)],
                        "deck": {
                            "player": _deck_info(decks[player]),
                            "opponent": _deck_info(decks[1 - player]),
                        },
                        "quality": {
                            "complete_game_trace": True,
                            "visible_observation_only": True,
                            "teacher_verified": False,
                            "action_aligned_from_next_record": True,
                            "forced_single_option": (
                                len(parsed.select.options) == 1
                                and parsed.select.min_count == 1
                                and parsed.select.max_count == 1
                            ),
                        },
                    }
                    sample_errors = validate_sample(row)
                    if sample_errors:
                        errors.append({"game_id": game_id, "player": player, "step": replay_step, "errors": sample_errors})
                    else:
                        rows.append(row)
                        stats["aligned_actions"] += 1
                    pending[player] = None
                except Exception as exc:
                    errors.append({"game_id": game_id, "player": player, "step": replay_step, "errors": [f"{type(exc).__name__}: {exc}"]})
                    pending[player] = None
            elif previous is not None and action not in (None, []):
                stats["unaligned_nonempty_actions"] += 1

            observation = record.get("observation")
            if record.get("status") == "ACTIVE" and isinstance(observation, dict) and isinstance(observation.get("select"), dict):
                pending[player] = observation
                stats["active_selects"] += 1
    stats["unresolved_pending"] += sum(item is not None for item in pending)
    return rows, errors, stats


def _paths(args: argparse.Namespace) -> Iterable[Path]:
    valid = None
    if args.index.exists():
        index = _load(args.index)
        valid = {
            Path(item["path"]).name
            for item in index.get("replays") or []
            if item.get("valid_complete_trajectory")
        }
    files = sorted(args.input.glob("episode-*-replay.json"))
    if valid is not None:
        files = [path for path in files if path.name in valid]
    return files[: args.max_files] if args.max_files else files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/external/kaggle_replays/raw")
    parser.add_argument("--index", type=Path, default=ROOT / "data/external/kaggle_replays/replay_index.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/kaggle_decisions.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/processed/kaggle_conversion_summary.json")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    tags = load_card_tags()
    files = list(_paths(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = Counter()
    errors: list[dict[str, Any]] = []
    games = Counter()
    sample_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for path in files:
            rows, replay_errors, stats = convert_replay(path, tags)
            total.update(stats)
            errors.extend(replay_errors)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                sample_count += 1
                games[row["split"]] += int(games[(row["split"], row["game_id"])] == 0)
                games[(row["split"], row["game_id"])] += 1
            # The threshold is checked only after the whole replay has been
            # written, so one game is never truncated across dataset builds.
            if args.max_samples and sample_count >= args.max_samples:
                break
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_files_considered": len(files),
        "converted_samples": sample_count,
        "converted_games": sum(games[name] for name in ("train", "valid", "test")),
        "split_games": {name: games[name] for name in ("train", "valid", "test")},
        "alignment": dict(total),
        "error_count": len(errors),
        "teacher_status": "observed_kaggle_agent_actions_only",
        "errors": errors[:100],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if sample_count and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
