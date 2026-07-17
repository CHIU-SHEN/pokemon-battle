#!/usr/bin/env python3
"""Build leakage-safe history features and future Combo weak targets."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "combo_labels_v1"
ACTION_FLAGS = {
    "play": 7,
    "attach": 8,
    "evolve": 9,
    "ability": 10,
    "retreat": 12,
    "attack": 13,
    "end": 14,
}
TAG_FLAGS = {
    "search": {"search_pokemon", "search_energy"},
    "draw": {"draw", "hand_refresh"},
    "setup": {"basic_pokemon", "setup_piece"},
    "evolution_piece": {"evolution_pokemon"},
    "energy": {"basic_energy", "attach_energy", "search_energy"},
    "switch": {"switch", "gust"},
    "attacker": {"main_attacker", "attacker"},
}
MILESTONES = tuple(ACTION_FLAGS) + tuple(TAG_FLAGS)
HISTORY_FEATURES = tuple(
    [f"past8_{name}" for name in MILESTONES]
    + [f"turn_{name}" for name in ACTION_FLAGS]
    + ["steps_since_attach", "steps_since_evolve", "steps_since_attack"]
)
COMBO_NAMES = (
    "setup_then_evolve",
    "search_then_play_or_evolve",
    "energy_then_attack",
    "switch_then_attack",
    "evolve_then_attack",
)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_at(stream, offset: int, byte_length: int) -> dict[str, Any]:
    stream.seek(offset)
    encoded = stream.read(byte_length)
    if len(encoded) != byte_length:
        raise ValueError(f"short read at {offset}")
    return json.loads(encoded)


def selected_milestones(row: dict[str, Any]) -> dict[str, bool]:
    options = row.get("options") or []
    chosen = [options[index] for index in row.get("observed_action") or [] if 0 <= int(index) < len(options)]
    option_types = {int(option.get("option_type", -1)) for option in chosen}
    tags = {str(tag) for option in chosen for tag in option.get("tags") or []}
    result = {name: value in option_types for name, value in ACTION_FLAGS.items()}
    result.update({name: bool(tags & accepted) for name, accepted in TAG_FLAGS.items()})
    return result


def history_vector(history: deque[dict[str, bool]], turn_actions: Counter[str], since: dict[str, int]) -> list[float]:
    values = [sum(float(item[name]) for item in history) / 8.0 for name in MILESTONES]
    values.extend(min(float(turn_actions[name]), 8.0) / 8.0 for name in ACTION_FLAGS)
    values.extend(min(float(since[name]), 32.0) / 32.0 for name in ("attach", "evolve", "attack"))
    return values


def future_flags(events: list[dict[str, bool]], position: int, horizon: int) -> dict[str, bool]:
    future = events[position + 1 : position + 1 + horizon]
    return {name: any(item[name] for item in future) for name in MILESTONES}


def combo_flags(current: dict[str, bool], future: dict[str, bool]) -> dict[str, bool]:
    return {
        "setup_then_evolve": (current["setup"] or current["search"]) and future["evolve"],
        "search_then_play_or_evolve": current["search"] and (future["play"] or future["evolve"]),
        "energy_then_attack": (current["energy"] or current["attach"]) and future["attack"],
        "switch_then_attack": (current["switch"] or current["retreat"]) and future["attack"],
        "evolve_then_attack": current["evolve"] and future["attack"],
    }


def build_labels(data_path: Path, index_path: Path, index_manifest_path: Path,
                 output_path: Path, manifest_path: Path) -> dict[str, Any]:
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    with data_path.open("rb") as data, index_path.open("r", encoding="utf-8") as index, output_path.open("wb") as output:
        for line in index:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            rows = [read_at(data, int(offset), int(length)) for offset, length in zip(
                trajectory["offsets"], trajectory["byte_lengths"]
            )]
            events = [selected_milestones(row) for row in rows]
            past: deque[dict[str, bool]] = deque(maxlen=8)
            turn_actions: Counter[str] = Counter()
            since = {"attach": 32, "evolve": 32, "attack": 32}
            previous_turn = None
            for position, (row, current) in enumerate(zip(rows, events)):
                turn = int(row["turn"])
                if previous_turn is None or turn != previous_turn:
                    turn_actions.clear()
                vector = history_vector(past, turn_actions, since)
                future8 = future_flags(events, position, 8)
                future16 = future_flags(events, position, 16)
                combos = combo_flags(current, future16)
                document = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": row["sample_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "game_id": trajectory["game_id"],
                    "player": trajectory["player"],
                    "split": trajectory["split"],
                    "position": position,
                    "turn": turn,
                    "history_features": vector,
                    "current_milestones": [name for name in MILESTONES if current[name]],
                    "future_8": [name for name in MILESTONES if future8[name]],
                    "future_16": [name for name in MILESTONES if future16[name]],
                    "combo_targets": [name for name in COMBO_NAMES if combos[name]],
                }
                encoded = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(encoded)
                digest.update(encoded)
                counts["samples"] += 1
                counts[f"split:{trajectory['split']}"] += 1
                for name in MILESTONES:
                    counts[f"current:{name}"] += current[name]
                    counts[f"future8:{name}"] += future8[name]
                    counts[f"future16:{name}"] += future16[name]
                for name in COMBO_NAMES:
                    counts[f"combo:{name}"] += combos[name]
                past.append(current)
                for name in ACTION_FLAGS:
                    turn_actions[name] += current[name]
                for name in since:
                    since[name] = 0 if current[name] else min(since[name] + 1, 32)
                previous_turn = turn

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": display_path(data_path),
        "trajectory_index": display_path(index_path),
        "trajectory_index_sha256": index_manifest["output_sha256"],
        "output": display_path(output_path),
        "output_sha256": digest.hexdigest().upper(),
        "samples": counts["samples"],
        "splits": {name: counts[f"split:{name}"] for name in ("train", "valid", "test")},
        "history_feature_names": list(HISTORY_FEATURES),
        "history_feature_count": len(HISTORY_FEATURES),
        "history_input_contract": "past/current-turn actions only; no future target is included",
        "milestone_names": list(MILESTONES),
        "current_milestones": {name: counts[f"current:{name}"] for name in MILESTONES},
        "future_8": {name: counts[f"future8:{name}"] for name in MILESTONES},
        "future_16": {name: counts[f"future16:{name}"] for name in MILESTONES},
        "combo_names": list(COMBO_NAMES),
        "combo_targets": {name: counts[f"combo:{name}"] for name in COMBO_NAMES},
        "target_contract": "future_8, future_16 and combo_targets are auxiliary labels only and must never enter model inputs",
        "ok": counts["samples"] == int(index_manifest["samples"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data/training/sequence_trajectories_v1.jsonl")
    parser.add_argument("--index-manifest", type=Path, default=ROOT / "data/training/sequence_manifest_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/combo_labels_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/combo_manifest_v1.json")
    args = parser.parse_args(argv)
    manifest = build_labels(args.data, args.index, args.index_manifest, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
