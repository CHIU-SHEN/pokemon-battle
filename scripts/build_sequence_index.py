#!/usr/bin/env python3
"""Audit training trajectories and build a compact short-sequence index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, BinaryIO, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "sequence_trajectory_index_v1"
DEFAULT_WINDOWS = (8, 16, 32)


def parse_windows(value: str) -> tuple[int, ...]:
    windows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not windows or any(length <= 0 for length in windows):
        raise argparse.ArgumentTypeError("window lengths must be positive integers")
    return windows


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def compact_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def iter_rows(stream: BinaryIO) -> Iterator[tuple[int, int, dict[str, Any]]]:
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line:
            return
        if line.strip():
            yield offset, len(line), json.loads(line)


def trajectory_id(game_id: str, player: int) -> str:
    digest = hashlib.sha256(f"{game_id}\0{player}".encode("utf-8")).hexdigest()[:20]
    return f"traj-{digest}-p{player}"


def build_index(
    data_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_manifest_path: Path,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_sha = str(source_manifest.get("sha256", "")).upper()
    trajectories: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    game_splits: dict[str, set[str]] = defaultdict(set)
    game_steps: dict[str, set[int]] = defaultdict(set)
    sample_ids: set[str] = set()
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    last_physical: dict[tuple[str, int], dict[str, Any]] = {}

    with data_path.open("rb") as stream:
        for offset, byte_length, row in iter_rows(stream):
            counts["samples"] += 1
            sample_id = str(row.get("sample_id", ""))
            game_id = str(row.get("game_id", ""))
            split = str(row.get("split", ""))
            try:
                player = int(row["current_player"])
                step = int(row["step"])
                turn = int(row["turn"])
            except (KeyError, TypeError, ValueError):
                errors.append({"sample_id": sample_id, "error": "invalid player/step/turn"})
                continue
            if not sample_id or not game_id or split not in {"train", "valid", "test"} or player not in {0, 1}:
                errors.append({"sample_id": sample_id, "error": "invalid identity/split/player"})
                continue
            if sample_id in sample_ids:
                errors.append({"sample_id": sample_id, "error": "duplicate sample_id"})
            sample_ids.add(sample_id)
            game_splits[game_id].add(split)
            if step in game_steps[game_id]:
                errors.append({"sample_id": sample_id, "error": "duplicate step within game", "step": step})
            game_steps[game_id].add(step)

            history = row.get("public_history") or []
            if not isinstance(history, list):
                errors.append({"sample_id": sample_id, "error": "public_history is not a list"})
                history = []
            visible = bool((row.get("quality") or {}).get("visible_observation_only", False))
            counts["visible_observation_only"] += visible
            counts[f"split:{split}"] += 1
            key = (game_id, player)
            prior = last_physical.get(key)
            if prior is not None:
                if step <= prior["step"]:
                    counts["physical_order_regressions"] += 1
                elif len(history) < prior["history_length"]:
                    counts["history_length_regressions"] += 1
                else:
                    counts["history_prefix_checks"] += 1
                    if compact_digest(history[: prior["history_length"]]) != prior["history_digest"]:
                        counts["history_prefix_mismatches"] += 1
            last_physical[key] = {
                "step": step,
                "history_length": len(history),
                "history_digest": compact_digest(history),
            }
            trajectories[key].append({
                "sample_id": sample_id,
                "offset": offset,
                "byte_length": byte_length,
                "step": step,
                "turn": turn,
                "history_length": len(history),
                "visible_observation_only": visible,
                "split": split,
            })

    cross_split_games = sorted(game_id for game_id, splits in game_splits.items() if len(splits) != 1)
    for game_id in cross_split_games[:100]:
        errors.append({"game_id": game_id, "error": "game crosses splits", "splits": sorted(game_splits[game_id])})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_digest = hashlib.sha256()
    length_histogram: Counter[int] = Counter()
    window_counts = {str(length): 0 for length in windows}
    full_window_counts = {str(length): 0 for length in windows}
    sorted_turn_regressions = 0
    with output_path.open("wb") as output:
        for (game_id, player), samples in sorted(trajectories.items()):
            physically_sorted = all(a["step"] < b["step"] for a, b in zip(samples, samples[1:]))
            samples.sort(key=lambda item: (item["step"], item["sample_id"]))
            sorted_turn_regressions += sum(a["turn"] > b["turn"] for a, b in zip(samples, samples[1:]))
            splits = sorted({item.pop("split") for item in samples})
            split = splits[0] if len(splits) == 1 else "invalid"
            length_histogram[len(samples)] += 1
            for length in windows:
                window_counts[str(length)] += len(samples)
                full_window_counts[str(length)] += max(0, len(samples) - length + 1)
            document = {
                "schema_version": SCHEMA_VERSION,
                "trajectory_id": trajectory_id(game_id, player),
                "game_id": game_id,
                "player": player,
                "split": split,
                "sample_count": len(samples),
                "physically_sorted": physically_sorted,
                "sample_ids": [item["sample_id"] for item in samples],
                "offsets": [item["offset"] for item in samples],
                "byte_lengths": [item["byte_length"] for item in samples],
                "steps": [item["step"] for item in samples],
                "turns": [item["turn"] for item in samples],
                "history_lengths": [item["history_length"] for item in samples],
                "visible_observation_only": [item["visible_observation_only"] for item in samples],
            }
            encoded = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            output.write(encoded)
            index_digest.update(encoded)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": display_path(data_path),
        "source_manifest": display_path(source_manifest_path),
        "source_sha256": expected_sha,
        "output": display_path(output_path),
        "output_sha256": index_digest.hexdigest().upper(),
        "samples": counts["samples"],
        "unique_sample_ids": len(sample_ids),
        "games": len(game_splits),
        "trajectories": len(trajectories),
        "splits": {name: counts[f"split:{name}"] for name in ("train", "valid", "test")},
        "window_lengths": list(windows),
        "left_padded_windows": window_counts,
        "full_windows": full_window_counts,
        "trajectory_length": {
            "min": min(length_histogram, default=0),
            "max": max(length_histogram, default=0),
            "histogram": {str(key): value for key, value in sorted(length_histogram.items())},
        },
        "audit": {
            "cross_split_games": cross_split_games,
            "physical_order_regressions": counts["physical_order_regressions"],
            "sorted_turn_regressions": sorted_turn_regressions,
            "visible_observation_only": counts["visible_observation_only"],
            "visible_observation_failures": counts["samples"] - counts["visible_observation_only"],
            "history_prefix_checks": counts["history_prefix_checks"],
            "history_length_regressions": counts["history_length_regressions"],
            "history_prefix_mismatches": counts["history_prefix_mismatches"],
            "public_history_contract": "visible event chunk attached to one observation; not a cumulative trajectory",
            "note": (
                "Length regressions and prefix mismatches confirm that public_history must not be treated as a cumulative prefix. "
                "Sequence order comes only from explicit game/player/step fields; each visible event chunk stays attached to its own observation."
            ),
        },
        "errors": errors[:100],
        "ok": (
            counts["samples"] > 0
            and counts["samples"] == int(source_manifest.get("samples", -1))
            and len(sample_ids) == counts["samples"]
            and not cross_split_games
            and not errors
            and sorted_turn_regressions == 0
            and counts["visible_observation_only"] == counts["samples"]
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/training/training_manifest_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/sequence_trajectories_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/sequence_manifest_v1.json")
    parser.add_argument("--windows", type=parse_windows, default=DEFAULT_WINDOWS)
    args = parser.parse_args(argv)
    manifest = build_index(args.data, args.output, args.manifest, args.source_manifest, args.windows)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
