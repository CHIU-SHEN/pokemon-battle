#!/usr/bin/env python3
"""Train a Top2 adapter from belief-PUCT visit targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arena.adapter_agent import AdapterArenaAgent  # noqa: E402
from src.rl.mcts_teacher import (  # noqa: E402
    TeacherConvergenceConfig,
    adapt_kl_coefficient,
    ema,
    evaluate_teacher_stop,
    gradient_norm,
    is_safe_checkpoint,
    relative_parameter_update,
    snapshot_parameters,
)
from src.rl.mcts_train import (  # noqa: E402
    collate_mcts_rows,
    evaluate_mcts_rows,
    load_mcts_rows,
    mcts_loss,
)
from src.train.shared_data import move_batch  # noqa: E402


def configure_teacher_parameters(
    model: torch.nn.Module,
) -> tuple[list[str], list[torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    allowed = ("adapter", "policy_delta", "value_delta")
    for module_name in allowed:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
    named = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise ValueError("teacher has no trainable parameters")
    return [name for name, _ in named], [parameter for _, parameter in named]


def validate_resume_identity(checkpoint: dict, expected: dict[str, str]) -> None:
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("teacher checkpoint identity mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/top2_rl_policy.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("primary", "reserve"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-wall-seconds", type=float, default=21600.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=1800.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--relative-update-max", type=float, default=1e-5)
    parser.add_argument("--policy-improvement-max", type=float, default=0.002)
    parser.add_argument("--value-worsening-max", type=float, default=0.01)
    parser.add_argument("--reference-kl-max", type=float, default=0.03)
    parser.add_argument("--convergence-patience", type=int, default=5)
    parser.add_argument("--min-convergence-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def load_adapter_state(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") not in {
        "top2_ppo_checkpoint_v1",
        "top2_mcts_checkpoint_v1",
        "top2_mcts_teacher_checkpoint_v2",
    }:
        raise ValueError("unsupported initial checkpoint")
    state = checkpoint["adapter_state"]
    model.adapter.load_state_dict(state["adapter"], strict=True)
    model.policy_delta.load_state_dict(state["policy_delta"], strict=True)
    model.value_delta.load_state_dict(state["value_delta"], strict=True)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.resume and args.initial_checkpoint:
        raise ValueError("resume and initial-checkpoint are mutually exclusive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    branch = next(item for item in config["branches"] if item["role"] == args.branch)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    owner = AdapterArenaAgent(
        branch["candidate_id"],
        project_root=args.project_root.resolve(),
        device=device,
    )
    model = owner.model.to(device)
    if args.initial_checkpoint:
        load_adapter_state(model, args.initial_checkpoint.resolve())
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    identity = {
        "branch": args.branch,
        "candidate_id": branch["candidate_id"],
        "deck_id": branch["deck_id"],
    }
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        if resume_checkpoint.get("schema_version") != "top2_mcts_teacher_checkpoint_v2":
            raise ValueError("unsupported teacher resume checkpoint")
        validate_resume_identity(resume_checkpoint, identity)
        load_adapter_state(model, args.resume.resolve())
        reference_state = resume_checkpoint["reference_adapter_state"]
        reference.adapter.load_state_dict(reference_state["adapter"], strict=True)
        reference.policy_delta.load_state_dict(reference_state["policy_delta"], strict=True)
        reference.value_delta.load_state_dict(reference_state["value_delta"], strict=True)
    train_rows = load_mcts_rows(
        args.samples.resolve(),
        branch=args.branch,
        deck_id=branch["deck_id"],
        split="train",
    )
    valid_rows = load_mcts_rows(
        args.samples.resolve(),
        branch=args.branch,
        deck_id=branch["deck_id"],
        split="valid",
    )
    trainable_names, parameters = configure_teacher_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        random.setstate(resume_checkpoint["python_random_state"])
        torch.set_rng_state(resume_checkpoint["torch_random_state"].cpu())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = list(resume_checkpoint.get("history", [])) if resume_checkpoint else []
    start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1 if resume_checkpoint else 1
    prior_elapsed = float(resume_checkpoint.get("elapsed_seconds", 0.0)) if resume_checkpoint else 0.0
    started = time.perf_counter()
    last_checkpoint_at = started
    relative_update_ema = (
        float(metrics[-1]["relative_update_ema"])
        if metrics and "relative_update_ema" in metrics[-1]
        else None
    )
    current_kl_coef = float(
        resume_checkpoint.get("adaptive_kl_coef", args.kl_coef)
        if resume_checkpoint else args.kl_coef
    )
    best_safe_metrics = resume_checkpoint.get("best_safe_metrics") if resume_checkpoint else None
    convergence = TeacherConvergenceConfig(
        relative_update_max=args.relative_update_max,
        policy_improvement_max=args.policy_improvement_max,
        value_worsening_max=args.value_worsening_max,
        reference_kl_max=args.reference_kl_max,
        patience=args.convergence_patience,
        max_wall_seconds=args.max_wall_seconds,
        min_convergence_seconds=args.min_convergence_seconds,
    )
    stop_decision = evaluate_teacher_stop(metrics, convergence, elapsed_seconds=prior_elapsed)
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        if stop_decision.stop:
            break
        random.Random(args.seed + epoch).shuffle(train_rows)
        sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "reference_kl": 0.0,
            "entropy": 0.0,
            "raw_grad_norm": 0.0,
            "clipped_grad_norm": 0.0,
            "relative_update": 0.0,
        }
        batches = 0
        model.train()
        for offset in range(0, len(train_rows), args.batch_size):
            raw = collate_mcts_rows(train_rows[offset: offset + args.batch_size])
            actions = raw.pop("actions")
            policy_targets = raw.pop("policy_targets")
            value_targets = raw.pop("value_targets").to(device)
            batch = move_batch(raw, device)
            output_doc = model(batch)
            with torch.no_grad():
                reference_doc = reference(batch)
            loss_doc = mcts_loss(
                logits=output_doc["policy_logits"],
                values=output_doc["value"],
                reference_logits=reference_doc["policy_logits"],
                actions=actions,
                policy_targets=policy_targets,
                value_targets=value_targets,
                legal_mask=batch["legal_mask"],
                value_coef=args.value_coef,
                kl_coef=current_kl_coef,
                entropy_coef=args.entropy_coef,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_doc["loss"].backward()
            sums["raw_grad_norm"] += gradient_norm(parameters)
            before = snapshot_parameters(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            sums["clipped_grad_norm"] += gradient_norm(parameters)
            optimizer.step()
            sums["relative_update"] += relative_parameter_update(before, parameters)
            for key in ("loss", "policy_loss", "value_loss", "reference_kl", "entropy"):
                sums[key] += float(loss_doc[key].detach().item())
            batches += 1
            elapsed = prior_elapsed + time.perf_counter() - started
            if args.max_batches and batches >= args.max_batches:
                break
            if args.max_wall_seconds > 0.0 and elapsed >= args.max_wall_seconds:
                break
            if (
                args.checkpoint_interval_seconds > 0.0
                and time.perf_counter() - last_checkpoint_at >= args.checkpoint_interval_seconds
            ):
                last_checkpoint_at = time.perf_counter()
        if batches == 0:
            raise ValueError("teacher epoch produced no batches")
        averaged = {key: value / batches for key, value in sums.items()}
        relative_update_ema = ema(relative_update_ema, averaged["relative_update"])
        holdout = evaluate_mcts_rows(
            model,
            reference,
            valid_rows,
            device=device,
            batch_size=args.batch_size,
            value_coef=args.value_coef,
            kl_coef=current_kl_coef,
            entropy_coef=args.entropy_coef,
        )
        elapsed = prior_elapsed + time.perf_counter() - started
        record = {
            "epoch": epoch,
            **averaged,
            "relative_update_ema": relative_update_ema,
            **{f"holdout_{key}": value for key, value in holdout.items()},
            "batches": batches,
            "samples": len(train_rows),
            "holdout_samples": len(valid_rows),
            "elapsed_seconds": elapsed,
            "kl_coef": current_kl_coef,
        }
        metrics.append(record)
        last_epoch = epoch
        stop_decision = evaluate_teacher_stop(metrics, convergence, elapsed_seconds=elapsed)
        checkpoint = {
            "schema_version": "top2_mcts_teacher_checkpoint_v2",
            **identity,
            "adapter_state": model.adapter_state_dict(),
            "reference_adapter_state": reference.adapter_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "history": metrics,
            "trainable_parameter_names": trainable_names,
            "effective_parameters": {
                "learning_rate": args.learning_rate,
                "value_coef": args.value_coef,
                "kl_coef": args.kl_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
            },
        }
        checkpoint["adaptive_kl_coef"] = current_kl_coef
        checkpoint["best_safe_metrics"] = best_safe_metrics
        atomic_torch_save(checkpoint, output / "last.pt")
        if is_safe_checkpoint(record, best_safe_metrics, reference_kl_max=args.reference_kl_max):
            best_safe_metrics = {
                key: float(record[key])
                for key in ("holdout_policy_loss", "holdout_value_loss", "holdout_reference_kl")
            }
            checkpoint["best_safe_metrics"] = best_safe_metrics
            atomic_torch_save(checkpoint, output / "best_safe.pt")
        print(json.dumps(record), flush=True)
        if stop_decision.stop:
            break
        current_kl_coef = adapt_kl_coefficient(
            float(record["holdout_reference_kl"]),
            current_kl_coef,
            hard_limit=args.reference_kl_max,
        )
    if not metrics:
        raise ValueError("teacher training produced no metrics")
    final = metrics[-1]
    eligible = not stop_decision.unsafe and all(
        math.isfinite(float(value))
        for value in final.values()
        if isinstance(value, float)
    )
    summary = {
        "schema_version": "top2_mcts_teacher_train_summary_v2",
        "eligible": eligible,
        "checkpoint": str(output / "best_safe.pt") if (output / "best_safe.pt").is_file() else None,
        "last_checkpoint": str(output / "last.pt"),
        **identity,
        "epochs_requested": args.epochs,
        "epochs_completed": last_epoch,
        "converged": stop_decision.converged,
        "time_limit_reached": stop_decision.reason == "wall_time_limit",
        "stop_reason": stop_decision.reason,
        "trainable_parameter_names": trainable_names,
        "wall_seconds": prior_elapsed + time.perf_counter() - started,
        "epochs": metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
