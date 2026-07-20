"""Streaming JSONL input and dynamic-option batching for SL-0-shared."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % max(buckets, 1)


class TrainingJsonlDataset(IterableDataset):
    """Read one split without loading the multi-GB dataset into memory.

    Rank and DataLoader-worker sharding both use sample_id hashing. This keeps
    every sample on exactly one consumer without requiring a random-access
    index. A bounded shuffle buffer provides useful stochasticity while
    retaining constant memory usage.
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        *,
        shuffle_buffer: int = 0,
        seed: int = 20260715,
        epoch: int = 0,
        max_samples: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"invalid split: {split}")
        self.path = Path(path)
        self.split = split
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.max_samples = max(0, int(max_samples))
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rows(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        shard_id = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        yielded = 0
        with _open_text(self.path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != self.split:
                    continue
                sample_id = str(row.get("sample_id", ""))
                if _stable_bucket(sample_id, shard_count) != shard_id:
                    continue
                yield row
                yielded += 1
                if self.max_samples and yielded >= self.max_samples:
                    return

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rows = self._rows()
        if self.shuffle_buffer <= 1:
            yield from rows
            return
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        buffer: list[dict[str, Any]] = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer


def _deck_ids(side: Any) -> list[int]:
    if not isinstance(side, dict):
        return []
    cards = side.get("cards")
    if not isinstance(cards, list):
        return []
    return [max(0, int(card_id)) for card_id in cards if isinstance(card_id, (int, float))]


def collate_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(rows)
    global_dim = len(rows[0]["features"])
    option_dim = len(rows[0]["option_features"][0]) if rows[0]["option_features"] else 0
    max_options = max(int(row["select"]["option_count"]) for row in rows)
    max_player_cards = max((len(_deck_ids((row.get("deck") or {}).get("player"))) for row in rows), default=0)
    max_opponent_cards = max((len(_deck_ids((row.get("deck") or {}).get("opponent"))) for row in rows), default=0)
    max_player_cards = max(max_player_cards, 1)
    max_opponent_cards = max(max_opponent_cards, 1)

    global_features = torch.zeros((batch_size, global_dim), dtype=torch.float32)
    option_features = torch.zeros((batch_size, max_options, option_dim), dtype=torch.float32)
    legal_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
    soft_policy = torch.zeros((batch_size, max_options), dtype=torch.float32)
    policy_weight = torch.zeros(batch_size, dtype=torch.float32)
    value_target = torch.zeros(batch_size, dtype=torch.float32)
    value_weight = torch.zeros(batch_size, dtype=torch.float32)
    player_deck = torch.zeros((batch_size, max_player_cards), dtype=torch.long)
    player_deck_mask = torch.zeros((batch_size, max_player_cards), dtype=torch.bool)
    opponent_deck = torch.zeros((batch_size, max_opponent_cards), dtype=torch.long)
    opponent_deck_mask = torch.zeros((batch_size, max_opponent_cards), dtype=torch.bool)
    sample_ids = []
    game_ids = []
    policy_sources = []
    forced_single_option = []
    history_dim = len(rows[0].get("history_features") or [])
    history_features = torch.zeros((batch_size, history_dim), dtype=torch.float32) if history_dim else None

    for batch_index, row in enumerate(rows):
        if len(row["features"]) != global_dim:
            raise ValueError("inconsistent global feature dimension")
        row_history = row.get("history_features") or []
        if len(row_history) != history_dim:
            raise ValueError("inconsistent history feature dimension")
        if history_features is not None:
            history_features[batch_index] = torch.tensor(row_history, dtype=torch.float32)
        count = int(row["select"]["option_count"])
        if count <= 0 or len(row["option_features"]) != count:
            raise ValueError(f"invalid option count for {row.get('sample_id')}")
        if any(len(option) != option_dim for option in row["option_features"]):
            raise ValueError("inconsistent option feature dimension")
        global_features[batch_index] = torch.tensor(row["features"], dtype=torch.float32)
        option_features[batch_index, :count] = torch.tensor(row["option_features"], dtype=torch.float32)
        row_legal = row.get("legal_mask") or [True] * count
        legal_mask[batch_index, :count] = torch.tensor(row_legal, dtype=torch.bool)
        target = torch.tensor(row["supervision"]["soft_policy"], dtype=torch.float32)
        soft_policy[batch_index, :count] = target
        head_weights = row["supervision"].get("head_weights") or {}
        # Empty actions have no explicit stop option in V1 and therefore must
        # not contribute a misleading all-zero cross-entropy target.
        policy_weight[batch_index] = 0.0 if float(target.sum()) <= 0 else float(head_weights.get("policy", 0.0))
        if row.get("value_target") is not None:
            value_target[batch_index] = float(row["value_target"])
            value_weight[batch_index] = float(head_weights.get("value", 0.0))

        pids = _deck_ids((row.get("deck") or {}).get("player"))
        oids = _deck_ids((row.get("deck") or {}).get("opponent"))
        if pids:
            player_deck[batch_index, : len(pids)] = torch.tensor(pids, dtype=torch.long)
            player_deck_mask[batch_index, : len(pids)] = True
        if oids:
            opponent_deck[batch_index, : len(oids)] = torch.tensor(oids, dtype=torch.long)
            opponent_deck_mask[batch_index, : len(oids)] = True
        sample_ids.append(row["sample_id"])
        game_ids.append(row["game_id"])
        policy_sources.append(str((row.get("supervision") or {}).get("policy_source") or "unknown"))
        forced_single_option.append(bool((row.get("quality") or {}).get("forced_single_option", count == 1)))

    if not legal_mask.any(dim=1).all():
        raise ValueError("every sample must expose at least one legal option")
    result = {
        "global_features": global_features,
        "option_features": option_features,
        "legal_mask": legal_mask,
        "soft_policy": soft_policy,
        "policy_weight": policy_weight,
        "value_target": value_target,
        "value_weight": value_weight,
        "player_deck": player_deck,
        "player_deck_mask": player_deck_mask,
        "opponent_deck": opponent_deck,
        "opponent_deck_mask": opponent_deck_mask,
        "sample_ids": sample_ids,
        "game_ids": game_ids,
        "policy_sources": policy_sources,
        "forced_single_option": forced_single_option,
    }
    if history_features is not None:
        result["history_features"] = history_features
    return result


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors in ordinary and nested model batches to one device."""
    return {
        key: (
            move_batch(value, device)
            if isinstance(value, dict)
            else value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }
