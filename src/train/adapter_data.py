"""Streaming, tier-weighted data for one Top10 Adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from src.train.shared_data import _open_text, _stable_bucket


class AdapterJsonlDataset(IterableDataset):
    def __init__(self, paths: list[Path], view_path: Path, split: str, *, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.paths = paths
        self.view = json.loads(view_path.read_text(encoding="utf-8"))
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.exact = {row["deck_sha256_sorted_ids"] for row in self.view["tiers"]["exact"]}
        self.similar = {row["deck_sha256_sorted_ids"] for row in self.view["tiers"]["similar"]}
        coverage = self.view["coverage"]
        target = self.view["sampling"]["target_mix"]
        self.weights = {}
        for tier in ("exact", "similar", "general"):
            count = int(coverage[tier]["samples"].get(split, 0))
            self.weights[tier] = float(target[tier]) / count if count else 0.0
        positive = [value for value in self.weights.values() if value > 0]
        scale = min(positive) if positive else 1.0
        self.weights = {key: value / scale for key, value in self.weights.items()}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        shard_id = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        for path in self.paths:
            with _open_text(path) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("split") != self.split or _stable_bucket(str(row.get("sample_id", "")), shard_count) != shard_id:
                        continue
                    player = (row.get("deck") or {}).get("player") or {}
                    deck_hash = str(player.get("sha256_sorted_ids") or "").lower()
                    if not deck_hash:
                        continue
                    tier = "exact" if deck_hash in self.exact else "similar" if deck_hash in self.similar else "general"
                    weight = self.weights[tier]
                    if weight <= 0:
                        continue
                    row = copy.deepcopy(row)
                    heads = row["supervision"].setdefault("head_weights", {})
                    heads["policy"] = float(heads.get("policy", 0.0)) * weight
                    heads["value"] = float(heads.get("value", 0.0)) * weight
                    row["adapter_tier"] = tier
                    yield row
