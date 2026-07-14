#!/usr/bin/env python3
"""Select high-value M6 samples for search reanalysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.distill_schema import load_jsonl  # noqa: E402


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(float(np.sum(exp)), 1e-9)


def _linear_scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    clean_x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    clean_w = np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.sum(clean_x * clean_w.reshape(1, -1), axis=1) + bias


def score_samples(samples: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    weights = np.asarray(model["policy_weights"], dtype=np.float64)
    bias = float(model.get("policy_bias", 0.0))
    ranked: list[dict[str, Any]] = []
    for sample in samples:
        x = np.asarray(sample["option_features"], dtype=np.float64)
        if len(x) == 0:
            continue
        logits = _linear_scores(x, weights, bias)
        probs = _softmax(logits)
        model_top = int(np.argmax(logits))
        final = set(sample.get("final_action", []))
        v0 = tuple(sample.get("v0_action", []))
        final_tuple = tuple(sample.get("final_action", []))
        confidence = float(np.max(probs))
        wrong_high_conf = confidence >= 0.65 and model_top not in final
        endgame = sample.get("turn", 0) >= 8 or sample.get("select", {}).get("option_count", 0) >= 8
        loss_result = sample.get("game_result") in {-1, 2}

        priority = 0.0
        priority += 3.0 if v0 != final_tuple else 0.0
        priority += 2.0 if sample.get("is_key") else 0.0
        priority += 2.0 if wrong_high_conf else 0.0
        priority += 1.0 if endgame else 0.0
        priority += 1.0 if loss_result else 0.0
        priority += confidence
        priority += 0.5 * float(sample.get("risk_target") or 0.0)

        ranked.append(
            {
                "sample_id": sample["sample_id"],
                "game_id": sample["game_id"],
                "priority": priority,
                "reason": {
                    "v0_v1_disagree": v0 != final_tuple,
                    "is_key": bool(sample.get("is_key")),
                    "wrong_high_conf": wrong_high_conf,
                    "endgame_or_many_options": endgame,
                    "loss_result": loss_result,
                },
                "model_top1": model_top,
                "model_confidence": confidence,
                "final_action": sample.get("final_action", []),
                "v0_action": sample.get("v0_action", []),
                "select": sample.get("select", {}),
                "search": sample.get("search", {}),
            }
        )
    ranked.sort(key=lambda row: row["priority"], reverse=True)
    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "samples.jsonl")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "model.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "dev_smoke" / "reanalysis_queue.json")
    parser.add_argument("--max-items", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples = load_jsonl(args.data)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    ranked = score_samples(samples, model)[: args.max_items]
    output = {
        "schema_version": "m6_reanalysis_queue_v1",
        "source_data": str(args.data),
        "source_model": str(args.model),
        "selection_rule": "prioritize V0/V1 disagreement, key/endgame states, and high-confidence model mistakes",
        "items": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"items": len(ranked), "top_priority": ranked[0]["priority"] if ranked else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
