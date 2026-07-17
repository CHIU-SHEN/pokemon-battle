"""Checks for deterministic, perspective-safe short-sequence indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from scripts.build_sequence_index import build_index
from src.train.sequence_data import SequenceWindowDataset, collate_sequence_windows


def row(sample: str, game: str, player: int, step: int, turn: int, history: list[dict]) -> dict:
    return {
        "sample_id": sample,
        "game_id": game,
        "split": "train",
        "current_player": player,
        "step": step,
        "turn": turn,
        "public_history": history,
        "features": [float(step), float(turn)],
        "select": {"option_count": 2},
        "option_features": [[0.0, 1.0], [1.0, 0.0]],
        "legal_mask": [True, True],
        "deck": {"player": {"cards": [1]}, "opponent": {"cards": [2]}},
        "value_target": 1.0,
        "supervision": {
            "soft_policy": [1.0, 0.0],
            "policy_source": "test",
            "head_weights": {"policy": 1.0, "value": 0.5},
        },
        "quality": {"visible_observation_only": True},
    }


def main() -> int:
    rows = [
        row("g:2", "g", 0, 2, 1, [{"type": 1}, {"type": 2}]),
        row("g:0", "g", 0, 0, 0, []),
        row("g:1", "g", 1, 1, 0, [{"type": 1}]),
        row("g:3", "g", 0, 3, 1, [{"type": 1}, {"type": 2}, {"type": 3}]),
    ]
    with tempfile.TemporaryDirectory(prefix="sequence_index_") as tmp:
        root = Path(tmp)
        data = root / "data.jsonl"
        encoded = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows).encode()
        data.write_bytes(encoded)
        source_manifest = root / "source.json"
        source_manifest.write_text(json.dumps({
            "ok": True,
            "samples": len(rows),
            "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        }), encoding="utf-8")
        output = root / "index.jsonl"
        manifest_path = root / "manifest.json"
        manifest = build_index(data, output, manifest_path, source_manifest, (2, 3))
        assert manifest["ok"]
        assert manifest["trajectories"] == 2
        assert manifest["audit"]["physical_order_regressions"] == 1
        assert manifest["left_padded_windows"] == {"2": 4, "3": 4}
        assert manifest["full_windows"] == {"2": 2, "3": 1}
        indexed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        player_zero = next(item for item in indexed if item["player"] == 0)
        assert player_zero["steps"] == [0, 2, 3]
        assert player_zero["sample_ids"] == ["g:0", "g:2", "g:3"]
        with data.open("rb") as stream:
            stream.seek(player_zero["offsets"][1])
            restored = json.loads(stream.read(player_zero["byte_lengths"][1]))
        assert restored["sample_id"] == "g:2"
        windows = list(SequenceWindowDataset(data, output, "train", window_length=3))
        assert len(windows) == 4
        final = next(item for item in windows if item["player"] == 0 and item["end_position"] == 2)
        assert [item["sample_id"] for item in final["rows"]] == ["g:0", "g:2", "g:3"]
        early = next(item for item in windows if item["player"] == 0 and item["end_position"] == 0)
        batch = collate_sequence_windows([early, final])
        assert batch["valid_mask"].tolist() == [[False, False, True], [True, True, True]]
        assert batch["reset_mask"].tolist() == [[False, False, True], [True, False, False]]
        assert batch["turn_boundary"].tolist() == [[False, False, True], [True, True, False]]
        assert batch["sequence_positions"].tolist() == [[0, 2], [1, 0], [1, 1], [1, 2]]
    print("OK: sequence index sorting, perspective split, windows and offsets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
