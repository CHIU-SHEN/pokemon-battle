#!/usr/bin/env python3
"""Train a tiny NumPy policy/value/risk model for M6 distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.distill_schema import load_jsonl, split_by_game, validate_dataset  # noqa: E402
from src.train.features import FEATURE_NAMES, TAG_FEATURES  # noqa: E402


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(float(np.sum(exp)), 1e-9)


def _clip_params(*arrays: np.ndarray, limit: float = 20.0) -> None:
    for array in arrays:
        np.nan_to_num(array, copy=False, nan=0.0, posinf=limit, neginf=-limit)
        np.clip(array, -limit, limit, out=array)


def _linear_scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    clean_x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    clean_w = np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.sum(clean_x * clean_w.reshape(1, -1), axis=1) + bias


def _linear_value(x: np.ndarray, weights: np.ndarray, bias: float) -> float:
    clean_x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    clean_w = np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0)
    return float(np.sum(clean_x * clean_w) + bias)


def _policy_targets(sample: dict[str, Any]) -> np.ndarray | None:
    option_count = int(sample["select"]["option_count"])
    action = [idx for idx in sample.get("final_action", []) if 0 <= idx < option_count]
    if not action:
        return None
    target = np.zeros(option_count, dtype=np.float64)
    for idx in action:
        target[idx] = 1.0 / len(action)
    return target


def _sample_matrix(sample: dict[str, Any]) -> np.ndarray:
    return np.asarray(sample["option_features"], dtype=np.float64)


def _mean_vector(sample: dict[str, Any]) -> np.ndarray:
    x = _sample_matrix(sample)
    if len(x) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    action = [idx for idx in sample.get("final_action", []) if 0 <= idx < len(x)]
    if action:
        return x[action].mean(axis=0)
    return x.mean(axis=0)


def train_model(samples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    train, valid, split = split_by_game(samples, valid_ratio=args.valid_ratio, seed=args.seed)
    dim = len(FEATURE_NAMES)
    rng = np.random.default_rng(args.seed)
    policy_w = rng.normal(0.0, 0.01, size=dim)
    policy_b = 0.0
    value_w = np.zeros(dim, dtype=np.float64)
    risk_w = np.zeros(dim, dtype=np.float64)
    value_b = 0.0
    risk_b = 0.0

    l2 = args.l2
    for _epoch in range(args.epochs):
        order = rng.permutation(len(train))
        for sample_idx in order:
            sample = train[int(sample_idx)]
            x = _sample_matrix(sample)
            if len(x) == 0:
                continue
            target = _policy_targets(sample)
            if target is not None:
                logits = _linear_scores(x, policy_w, policy_b)
                probs = _softmax(logits)
                grad_logits = probs - target
                policy_grad = np.sum(x * grad_logits.reshape(-1, 1), axis=0)
                policy_w -= args.lr * (policy_grad + l2 * policy_w)
                policy_b -= args.lr * float(np.sum(grad_logits))
                policy_b = max(-20.0, min(20.0, policy_b))

            mean_x = _mean_vector(sample)
            if sample.get("value_target") is not None:
                value_target = float(sample["value_target"])
                value_pred = _linear_value(mean_x, value_w, value_b)
                err = value_pred - value_target
                value_w -= args.lr_value * (err * mean_x + l2 * value_w)
                value_b -= args.lr_value * err
            if sample.get("risk_target") is not None:
                risk_target = float(sample["risk_target"])
                risk_pred = _linear_value(mean_x, risk_w, risk_b)
                err = risk_pred - risk_target
                risk_w -= args.lr_value * (err * mean_x + l2 * risk_w)
                risk_b -= args.lr_value * err
            _clip_params(policy_w, value_w, risk_w)
            value_b = max(-5.0, min(5.0, value_b))
            risk_b = max(-5.0, min(5.0, risk_b))

    metrics = evaluate(policy_w, policy_b, value_w, value_b, risk_w, risk_b, train, valid)
    metrics["split"] = split
    model = {
        "model_version": "v2_policy_linear_numpy_v1",
        "created_by": "src/train/train_distill.py",
        "status": "experimental_not_promoted",
        "feature_names": FEATURE_NAMES,
        "tag_features": TAG_FEATURES,
        "policy_weights": policy_w.astype(float).tolist(),
        "policy_bias": float(policy_b),
        "value_weights": value_w.astype(float).tolist(),
        "value_bias": float(value_b),
        "risk_weights": risk_w.astype(float).tolist(),
        "risk_bias": float(risk_b),
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_value": args.lr_value,
            "l2": args.l2,
            "sample_count": len(samples),
        },
        "promotion_gate": {
            "ready": False,
            "reason": "Smoke-scale M6 dataset only; V1 search is stable but not yet proven stronger or large enough for production distillation.",
        },
    }
    return model, metrics


def _topk_metrics(policy_w: np.ndarray, policy_b: float, samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    top1 = 0
    top3 = 0
    ce_losses: list[float] = []
    for sample in samples:
        target = _policy_targets(sample)
        if target is None:
            continue
        x = _sample_matrix(sample)
        logits = _linear_scores(x, policy_w, policy_b)
        probs = _softmax(logits)
        ranking = list(np.argsort(-logits))
        target_indices = set(np.where(target > 0)[0].tolist())
        total += 1
        top1 += int(ranking[0] in target_indices)
        top3 += int(bool(target_indices.intersection(ranking[:3])))
        ce_losses.append(float(-np.sum(target * np.log(np.maximum(probs, 1e-9)))))
    return {
        "policy_samples": total,
        "policy_top1": top1 / total if total else None,
        "policy_top3": top3 / total if total else None,
        "policy_ce": sum(ce_losses) / len(ce_losses) if ce_losses else None,
    }


def _regression_metrics(weights: np.ndarray, bias: float, samples: list[dict[str, Any]], target_name: str) -> dict[str, Any]:
    ys = []
    preds = []
    for sample in samples:
        if sample.get(target_name) is None:
            continue
        x = _mean_vector(sample)
        preds.append(_linear_value(x, weights, bias))
        ys.append(float(sample[target_name]))
    if not ys:
        return {f"{target_name}_samples": 0, f"{target_name}_mse": None, f"{target_name}_corr": None}
    y = np.asarray(ys)
    p = np.asarray(preds)
    mse = float(np.mean((p - y) ** 2))
    corr = float(np.corrcoef(p, y)[0, 1]) if len(y) >= 2 and np.std(p) > 1e-9 and np.std(y) > 1e-9 else None
    return {f"{target_name}_samples": len(ys), f"{target_name}_mse": mse, f"{target_name}_corr": corr}


def _latency_ms(policy_w: np.ndarray, policy_b: float, samples: list[dict[str, Any]], repeats: int = 200) -> float | None:
    matrices = [_sample_matrix(sample) for sample in samples if sample.get("option_features")]
    if not matrices:
        return None
    started = time.perf_counter()
    count = 0
    for _ in range(repeats):
        for x in matrices:
            _ = _linear_scores(x, policy_w, policy_b)
            count += 1
    return (time.perf_counter() - started) * 1000.0 / max(count, 1)


def evaluate(
    policy_w: np.ndarray,
    policy_b: float,
    value_w: np.ndarray,
    value_b: float,
    risk_w: np.ndarray,
    risk_b: float,
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "train": _topk_metrics(policy_w, policy_b, train),
        "valid": _topk_metrics(policy_w, policy_b, valid),
        "latency_ms_per_select": _latency_ms(policy_w, policy_b, valid or train),
    }
    metrics["train"].update(_regression_metrics(value_w, value_b, train, "value_target"))
    metrics["valid"].update(_regression_metrics(value_w, value_b, valid, "value_target"))
    metrics["train"].update(_regression_metrics(risk_w, risk_b, train, "risk_target"))
    metrics["valid"].update(_regression_metrics(risk_w, risk_b, valid, "risk_target"))
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "samples.jsonl")
    parser.add_argument("--model-out", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "model.json")
    parser.add_argument("--metrics-out", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "train_metrics.json")
    parser.add_argument("--splits-out", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "splits.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--lr-value", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples = load_jsonl(args.data)
    validation = validate_dataset(samples)
    if not validation["ok"]:
        print(json.dumps(validation, ensure_ascii=False), file=sys.stderr)
        return 1
    model, metrics = train_model(samples, args)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps(metrics["split"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
