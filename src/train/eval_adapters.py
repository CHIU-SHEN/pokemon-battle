#!/usr/bin/env python3
"""Evaluate all deck adapters against their frozen SL-0 base on one test pass."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import pathlib
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.adapter_model import DeckAdapterPolicyValueNet  # noqa: E402
from src.train.shared_data import (  # noqa: E402
    TrainingJsonlDataset,
    collate_training_rows,
    move_batch,
)
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "artifacts/sl0_shared_full/best.pt",
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=ROOT / "artifacts/adapters_top10",
    )
    parser.add_argument(
        "--view-root",
        type=Path,
        default=ROOT / "data/adapter_views",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each training_decision_v1 JSONL. Defaults to the frozen base dataset.",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/top10_adapter_offline_eval.json",
    )
    return parser.parse_args(argv)


@contextmanager
def portable_checkpoint_paths():
    original = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        pathlib.PosixPath = original  # type: ignore[misc,assignment]


class CombinedDataset(IterableDataset):
    def __init__(self, paths: list[Path], split: str, max_samples: int = 0) -> None:
        super().__init__()
        self.paths = paths
        self.split = split
        self.max_samples = max_samples

    def __iter__(self):
        yielded = 0
        for path in self.paths:
            remaining = 0 if not self.max_samples else self.max_samples - yielded
            if self.max_samples and remaining <= 0:
                return
            for row in TrainingJsonlDataset(path, self.split, max_samples=remaining):
                yield row
                yielded += 1


def collate_with_deck_hash(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_training_rows(rows)
    batch["deck_hashes"] = [
        str((((row.get("deck") or {}).get("player") or {}).get("sha256_sorted_ids")) or "").lower()
        for row in rows
    ]
    return batch


class Totals:
    def __init__(self) -> None:
        self.rows = 0
        self.policy_count = 0
        self.policy_loss = 0.0
        self.policy_correct = 0
        self.value_count = 0
        self.value_squared_error = 0.0
        self.illegal_predictions = 0

    def add(
        self,
        *,
        policy_loss: float,
        policy_hit: bool | None,
        value_squared_error: float | None,
        illegal: bool,
    ) -> None:
        self.rows += 1
        self.illegal_predictions += int(illegal)
        if policy_hit is not None:
            self.policy_count += 1
            self.policy_loss += policy_loss
            self.policy_correct += int(policy_hit)
        if value_squared_error is not None:
            self.value_count += 1
            self.value_squared_error += value_squared_error

    def report(self) -> dict[str, int | float | None]:
        return {
            "rows": self.rows,
            "policy_count": self.policy_count,
            "policy_loss": self.policy_loss / self.policy_count if self.policy_count else None,
            "policy_top1": self.policy_correct / self.policy_count if self.policy_count else None,
            "value_count": self.value_count,
            "value_mse": (
                self.value_squared_error / self.value_count if self.value_count else None
            ),
            "illegal_predictions": self.illegal_predictions,
        }


def load_models(
    base_path: Path,
    adapter_root: Path,
    view_root: Path,
    device: torch.device,
) -> tuple[SharedPolicyValueNet, dict[str, DeckAdapterPolicyValueNet], dict[str, dict[str, set[str]]], dict]:
    with portable_checkpoint_paths():
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    if base_checkpoint.get("schema_version") != "sl0_shared_checkpoint_v1":
        raise ValueError(f"unsupported base schema: {base_checkpoint.get('schema_version')!r}")
    config = SharedModelConfig(**base_checkpoint["model_config"])
    base = SharedPolicyValueNet(config)
    base.load_state_dict(base_checkpoint["model_state"], strict=True)
    base.to(device).eval()

    adapters: dict[str, DeckAdapterPolicyValueNet] = {}
    tiers: dict[str, dict[str, set[str]]] = {}
    for directory in sorted(path for path in adapter_root.iterdir() if path.is_dir()):
        candidate = directory.name
        checkpoint_path = directory / "best.pt"
        view_path = view_root / candidate / "view.json"
        if not checkpoint_path.is_file() or not view_path.is_file():
            raise FileNotFoundError(f"missing checkpoint or view for {candidate}")
        with portable_checkpoint_paths():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "deck_adapter_checkpoint_v1":
            raise ValueError(f"unsupported adapter schema for {candidate}")
        if checkpoint.get("candidate_id") != candidate:
            raise ValueError(f"candidate mismatch for {candidate}")
        if str(checkpoint.get("base_dataset_sha256", "")).upper() != str(
            base_checkpoint.get("dataset_sha256", "")
        ).upper():
            raise ValueError(f"base dataset mismatch for {candidate}")
        model = DeckAdapterPolicyValueNet(base, int(checkpoint["bottleneck_dim"]))
        state = checkpoint["adapter_state"]
        model.adapter.load_state_dict(state["adapter"], strict=True)
        model.policy_delta.load_state_dict(state["policy_delta"], strict=True)
        model.value_delta.load_state_dict(state["value_delta"], strict=True)
        adapters[candidate] = model.to(device).eval()
        view = json.loads(view_path.read_text(encoding="utf-8"))
        tiers[candidate] = {
            name: {str(row["deck_sha256_sorted_ids"]).lower() for row in view["tiers"][name]}
            for name in ("exact", "similar")
        }
    return base, adapters, tiers, base_checkpoint


def metrics_for_batch(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, list[Any]]:
    logits = outputs["policy_logits"].float()
    predicted = logits.argmax(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    per_policy = -(batch["soft_policy"].float() * log_probs).sum(dim=-1)
    per_value = (outputs["value"].float() - batch["value_target"].float()).square()
    target_mask = batch["soft_policy"] > 0
    return {
        "policy_loss": per_policy.detach().cpu().tolist(),
        "policy_hit": target_mask.gather(1, predicted[:, None]).squeeze(1).detach().cpu().tolist(),
        "policy_active": (batch["policy_weight"] > 0).detach().cpu().tolist(),
        "value_squared_error": per_value.detach().cpu().tolist(),
        "value_active": (batch["value_weight"] > 0).detach().cpu().tolist(),
        "illegal": (
            ~batch["legal_mask"].gather(1, predicted[:, None]).squeeze(1)
        ).detach().cpu().tolist(),
    }


def add_batch(totals: Totals, metrics: dict[str, list[Any]], indices: list[int]) -> None:
    for index in indices:
        totals.add(
            policy_loss=float(metrics["policy_loss"][index]),
            policy_hit=bool(metrics["policy_hit"][index]) if metrics["policy_active"][index] else None,
            value_squared_error=(
                float(metrics["value_squared_error"][index])
                if metrics["value_active"][index]
                else None
            ),
            illegal=bool(metrics["illegal"][index]),
        )


def relative_delta(adapted: dict, base: dict) -> dict[str, float | None]:
    def delta(key: str) -> float | None:
        if adapted[key] is None or base[key] is None:
            return None
        return float(adapted[key]) - float(base[key])

    return {
        "policy_loss": delta("policy_loss"),
        "policy_top1": delta("policy_top1"),
        "value_mse": delta("value_mse"),
    }


def decision(candidate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    exact = candidate["tiers"]["exact"]
    similar = candidate["tiers"]["similar"]
    general = candidate["tiers"]["general"]
    if candidate["illegal_predictions"]:
        reasons.append("存在非法 top-1")
        return "reject", reasons
    if exact["rows"] < 100:
        reasons.append("exact test 样本不足 100，仅作观察")
    if exact["delta"]["policy_top1"] is not None and exact["delta"]["policy_top1"] < -0.005:
        reasons.append("exact top-1 回退超过 0.5pp")
    if similar["delta"]["policy_top1"] is not None and similar["delta"]["policy_top1"] < -0.005:
        reasons.append("similar top-1 回退超过 0.5pp")
    if general["delta"]["policy_top1"] is not None and general["delta"]["policy_top1"] < -0.003:
        reasons.append("general top-1 回退超过 0.3pp")
    if any("回退" in reason for reason in reasons):
        return "retrain", reasons
    exact_gain = exact["delta"]["policy_top1"] or 0.0
    similar_gain = similar["delta"]["policy_top1"] or 0.0
    exact_loss_gain = -(exact["delta"]["policy_loss"] or 0.0)
    if exact["rows"] >= 100 and (
        exact_gain >= 0.003 or exact_loss_gain >= 0.01 or similar_gain >= 0.002
    ):
        reasons.append("专属或相似牌组离线指标提升，且通用层未越过回退门槛")
        return "advance", reasons
    reasons.append("未观察到足够稳定的专属收益")
    return "hold", reasons


def benchmark_single_row(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    *,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float | int]:
    timings = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - tick) * 1000.0
            if index >= warmup:
                timings.append(elapsed_ms)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "repeats": repeats,
        "mean_ms": statistics.fmean(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": ordered[p95_index],
        "max_ms": max(timings),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0:
        raise ValueError("invalid batch/worker/sample setting")
    paths = args.data or [ROOT / "data/training/training_decisions_v1.jsonl"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    base, adapters, tiers, base_checkpoint = load_models(
        args.base_checkpoint, args.adapter_root, args.view_root, device
    )
    loader = DataLoader(
        CombinedDataset(paths, args.split, args.max_samples),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_deck_hash,
        pin_memory=device.type == "cuda",
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    base_totals = {
        candidate: {tier: Totals() for tier in ("exact", "similar", "general")}
        for candidate in adapters
    }
    adapter_totals = {
        candidate: {tier: Totals() for tier in ("exact", "similar", "general")}
        for candidate in adapters
    }
    latency_seconds = {"base": 0.0, **{candidate: 0.0 for candidate in adapters}}
    rows = batches = unknown_rows = 0
    benchmark_batch = None
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            deck_hashes = raw_batch.pop("deck_hashes")
            for key in ("sample_ids", "game_ids", "policy_sources", "forced_single_option"):
                raw_batch.pop(key, None)
            batch = move_batch(raw_batch, device)
            if benchmark_batch is None:
                benchmark_batch = {
                    key: value[:1]
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                base_outputs = base(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_seconds["base"] += time.perf_counter() - tick
            base_metrics = metrics_for_batch(base_outputs, batch)
            batch_indices: dict[str, dict[str, list[int]]] = {}
            for candidate in adapters:
                candidate_indices = {"exact": [], "similar": [], "general": []}
                exact, similar = tiers[candidate]["exact"], tiers[candidate]["similar"]
                for index, deck_hash in enumerate(deck_hashes):
                    if not deck_hash:
                        continue
                    tier = "exact" if deck_hash in exact else "similar" if deck_hash in similar else "general"
                    candidate_indices[tier].append(index)
                batch_indices[candidate] = candidate_indices
                for tier, indices in candidate_indices.items():
                    add_batch(base_totals[candidate][tier], base_metrics, indices)
            for candidate, model in adapters.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency_seconds[candidate] += time.perf_counter() - tick
                metrics = metrics_for_batch(outputs, batch)
                for tier, indices in batch_indices[candidate].items():
                    add_batch(adapter_totals[candidate][tier], metrics, indices)
            rows += len(deck_hashes)
            unknown_rows += sum(not deck_hash for deck_hash in deck_hashes)
            batches += 1
            if batches % 50 == 0:
                print(json.dumps({"rows": rows, "batches": batches}), flush=True)

    if benchmark_batch is None:
        raise ValueError("evaluation split produced no rows")
    online_latency = {
        "base": benchmark_single_row(base, benchmark_batch, device, amp_enabled)
    }
    for candidate, model in adapters.items():
        online_latency[candidate] = benchmark_single_row(
            model, benchmark_batch, device, amp_enabled
        )

    candidates = {}
    for candidate in adapters:
        tier_reports = {}
        illegal = 0
        for tier in ("exact", "similar", "general"):
            base_report = base_totals[candidate][tier].report()
            adapted_report = adapter_totals[candidate][tier].report()
            illegal += int(adapted_report["illegal_predictions"])
            tier_reports[tier] = {
                "base": base_report,
                "adapter": adapted_report,
                "delta": relative_delta(adapted_report, base_report),
                "rows": adapted_report["rows"],
            }
        item = {
            "tiers": tier_reports,
            "illegal_predictions": illegal,
            "latency_ms_per_row": latency_seconds[candidate] * 1000.0 / max(rows, 1),
            "latency_delta_ms_per_row": (
                latency_seconds[candidate] - latency_seconds["base"]
            ) * 1000.0 / max(rows, 1),
            "online_latency_batch1": online_latency[candidate],
        }
        item["decision"], item["decision_reasons"] = decision(item)
        candidates[candidate] = item
    report = {
        "schema_version": "top10_adapter_offline_eval_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": args.split,
        "data": [str(path) for path in paths],
        "rows": rows,
        "unknown_rows_excluded_from_tiers": unknown_rows,
        "batches": batches,
        "device": str(device),
        "amp": amp_enabled,
        "elapsed_seconds": time.perf_counter() - started,
        "base_checkpoint": str(args.base_checkpoint),
        "base_dataset_sha256": base_checkpoint.get("dataset_sha256"),
        "base_latency_ms_per_row": latency_seconds["base"] * 1000.0 / max(rows, 1),
        "base_online_latency_batch1": online_latency["base"],
        "decision_policy": {
            "hard_gate": "0 illegal predictions",
            "retrain": "exact/similar top-1 < -0.5pp or general top-1 < -0.3pp",
            "advance": "exact rows >= 100 and exact top-1 >= +0.3pp, exact policy loss <= -0.01, or similar top-1 >= +0.2pp",
            "note": "offline gate only; Arena remains mandatory",
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": rows, "candidates": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
