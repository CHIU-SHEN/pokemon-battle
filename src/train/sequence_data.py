"""Random-access short windows over the compact trajectory index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.train.shared_data import _stable_bucket, collate_training_rows
from src.train.transition_features import TRANSITION_DIM, previous_action_features, transition_features


class SequenceWindowDataset(IterableDataset):
    def __init__(
        self,
        data_path: str | Path,
        index_path: str | Path,
        split: str,
        *,
        window_length: int = 16,
        require_full_window: bool = False,
        max_windows: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"invalid split: {split}")
        if window_length <= 0 or max_windows < 0:
            raise ValueError("window_length must be positive and max_windows cannot be negative")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.split = split
        self.window_length = int(window_length)
        self.require_full_window = bool(require_full_window)
        self.max_windows = int(max_windows)
        self.rank = int(rank)
        self.world_size = int(world_size)

    @staticmethod
    def _read_row(stream, offset: int, byte_length: int) -> dict[str, Any]:
        stream.seek(offset)
        encoded = stream.read(byte_length)
        if len(encoded) != byte_length:
            raise ValueError(f"short read at offset {offset}: expected {byte_length}, got {len(encoded)}")
        return json.loads(encoded)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        shard_id = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        yielded = 0
        with self.data_path.open("rb") as data, self.index_path.open("r", encoding="utf-8") as index:
            for line in index:
                if not line.strip():
                    continue
                trajectory = json.loads(line)
                if trajectory.get("split") != self.split:
                    continue
                trajectory_id = str(trajectory["trajectory_id"])
                if _stable_bucket(trajectory_id, shard_count) != shard_id:
                    continue
                count = int(trajectory["sample_count"])
                offsets = trajectory["offsets"]
                byte_lengths = trajectory["byte_lengths"]
                turns = trajectory["turns"]
                if not (len(offsets) == len(byte_lengths) == len(turns) == count):
                    raise ValueError(f"corrupt trajectory columns: {trajectory_id}")
                first_endpoint = self.window_length - 1 if self.require_full_window else 0
                for endpoint in range(first_endpoint, count):
                    start = max(0, endpoint - self.window_length + 1)
                    previous_row = (
                        self._read_row(data, int(offsets[start - 1]), int(byte_lengths[start - 1]))
                        if start > 0 else None
                    )
                    rows = [
                        self._read_row(data, int(offsets[position]), int(byte_lengths[position]))
                        for position in range(start, endpoint + 1)
                    ]
                    yield {
                        "trajectory_id": trajectory_id,
                        "game_id": trajectory["game_id"],
                        "player": int(trajectory["player"]),
                        "split": self.split,
                        "window_length": self.window_length,
                        "valid_length": len(rows),
                        "start_position": start,
                        "end_position": endpoint,
                        "turns": [int(value) for value in turns[start : endpoint + 1]],
                        "rows": rows,
                        "previous_row": previous_row,
                    }
                    yielded += 1
                    if self.max_windows and yielded >= self.max_windows:
                        return


def collate_sequence_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    if not windows:
        raise ValueError("cannot collate empty sequence windows")
    length = int(windows[0]["window_length"])
    if any(int(window["window_length"]) != length for window in windows):
        raise ValueError("all sequence windows in a batch must have the same window_length")
    batch_size = len(windows)
    valid_mask = torch.zeros((batch_size, length), dtype=torch.bool)
    reset_mask = torch.zeros((batch_size, length), dtype=torch.bool)
    turn_boundary = torch.zeros((batch_size, length), dtype=torch.bool)
    flat_rows: list[dict[str, Any]] = []
    sequence_positions: list[tuple[int, int]] = []
    transition_rows: list[list[float]] = []
    previous_action_rows: list[list[float]] = []
    endpoint_flat_indices: list[int] = []
    for batch_index, window in enumerate(windows):
        rows = window["rows"]
        turns = window["turns"]
        valid_length = len(rows)
        if valid_length <= 0 or valid_length > length or valid_length != len(turns):
            raise ValueError("invalid sequence window length")
        padding = length - valid_length
        valid_mask[batch_index, padding:] = True
        reset_mask[batch_index, padding] = True
        prior = window.get("previous_row")
        option_dim = len(rows[0]["option_features"][0])
        for local_index, (row, turn) in enumerate(zip(rows, turns)):
            time_index = padding + local_index
            if local_index == 0 or int(turn) != int(turns[local_index - 1]):
                turn_boundary[batch_index, time_index] = True
            flat_rows.append(row)
            sequence_positions.append((batch_index, time_index))
            transition_rows.append(transition_features(prior, row))
            previous_action_rows.append(previous_action_features(prior, option_dim))
            prior = row
        endpoint_flat_indices.append(len(flat_rows) - 1)
    flat_batch = collate_training_rows(flat_rows)
    return {
        "flat_batch": flat_batch,
        "sequence_positions": torch.tensor(sequence_positions, dtype=torch.long),
        "transition_features": torch.tensor(transition_rows, dtype=torch.float32),
        "previous_action_features": torch.tensor(previous_action_rows, dtype=torch.float32),
        "endpoint_flat_indices": torch.tensor(endpoint_flat_indices, dtype=torch.long),
        "valid_mask": valid_mask,
        "reset_mask": reset_mask,
        "turn_boundary": turn_boundary,
        "trajectory_ids": [window["trajectory_id"] for window in windows],
        "game_ids": [window["game_id"] for window in windows],
        "players": torch.tensor([window["player"] for window in windows], dtype=torch.long),
        "end_positions": torch.tensor([window["end_position"] for window in windows], dtype=torch.long),
        "transition_dim": TRANSITION_DIM,
    }
