#!/usr/bin/env python3
"""Materialize the SL-0-history streaming dataset in trajectory order."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_combo_labels import ACTION_FLAGS, HISTORY_FEATURES, history_vector, read_at, selected_milestones  # noqa: E402


SCHEMA_VERSION = "training_decision_history_v1"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build(data_path: Path, index_path: Path, index_manifest_path: Path,
          output_path: Path, manifest_path: Path) -> dict:
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("rb") as data, index_path.open(encoding="utf-8") as index, output_path.open("wb") as output:
        for line in index:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            past: deque[dict[str, bool]] = deque(maxlen=8)
            turn_actions: Counter[str] = Counter()
            since = {"attach": 32, "evolve": 32, "attack": 32}
            previous_turn = None
            for offset, byte_length in zip(trajectory["offsets"], trajectory["byte_lengths"]):
                row = read_at(data, int(offset), int(byte_length))
                turn = int(row["turn"])
                if previous_turn is None or turn != previous_turn:
                    turn_actions.clear()
                row = dict(row)
                row["base_schema_version"] = row["schema_version"]
                row["schema_version"] = SCHEMA_VERSION
                row["history_features"] = history_vector(past, turn_actions, since)
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(encoded)
                digest.update(encoded)
                counts["samples"] += 1
                counts[f"split:{row['split']}"] += 1
                current = selected_milestones(row)
                past.append(current)
                for name in ACTION_FLAGS:
                    turn_actions[name] += current[name]
                for name in since:
                    since[name] = 0 if current[name] else min(since[name] + 1, 32)
                previous_turn = turn
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_data": display_path(data_path),
        "base_data_sha256": index_manifest["source_sha256"],
        "trajectory_index": display_path(index_path),
        "trajectory_index_sha256": index_manifest["output_sha256"],
        "output": display_path(output_path),
        "sha256": digest.hexdigest().upper(),
        "samples": counts["samples"],
        "splits": {name: counts[f"split:{name}"] for name in ("train", "valid", "test")},
        "history_feature_names": list(HISTORY_FEATURES),
        "history_dim": len(HISTORY_FEATURES),
        "leakage_contract": "history_features use prior same-player decisions and current-turn past actions only",
        "ok": counts["samples"] == int(index_manifest["samples"]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training/training_decisions_v1.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data/training/sequence_trajectories_v1.jsonl")
    parser.add_argument("--index-manifest", type=Path, default=ROOT / "data/training/sequence_manifest_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/training_decisions_history_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/training/training_history_manifest_v1.json")
    args = parser.parse_args(argv)
    manifest = build(args.data, args.index, args.index_manifest, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
