#!/usr/bin/env python3
"""Verify Top2 RL branch identities, frozen hashes, policy, and package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(root: Path, item: dict, path_key: str, hash_key: str) -> None:
    path = root / item[path_key]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual.lower() != str(item[hash_key]).lower():
        raise ValueError(f"SHA-256 mismatch: {item[path_key]} expected={item[hash_key]} actual={actual}")


def verify_policy_root(root: Path) -> dict:
    root = root.resolve()
    policy_path = root / "config/top2_rl_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "top2_rl_policy_v1":
        raise ValueError("unsupported Top2 RL policy schema")
    branches = policy.get("branches") or []
    roles = [branch.get("role") for branch in branches]
    if roles != ["primary", "reserve"]:
        raise ValueError(f"roles must be ordered primary/reserve: {roles}")
    if len({branch.get("candidate_id") for branch in branches}) != 2:
        raise ValueError("Top2 candidates must be distinct")
    if len({branch.get("deck_id") for branch in branches}) != 2:
        raise ValueError("Top2 deck_id streams must be distinct")
    primary_budget = float(branches[0].get("budget_weight", 0))
    reserve_budget = float(branches[1].get("budget_weight", 0))
    ratio = reserve_budget / primary_budget if primary_budget else 0.0
    if not 0.3 <= ratio <= 0.5:
        raise ValueError(f"reserve budget ratio must stay in [0.3, 0.5]: {ratio}")
    split = policy.get("split") or {}
    if [split.get("train_percent"), split.get("valid_percent"), split.get("test_percent")] != [80, 10, 10]:
        raise ValueError("split must be 80/10/10")
    if split.get("holdout_never_train_percent") != 20:
        raise ValueError("20% never-train holdout is required")
    rollout = policy.get("rollout") or {}
    if int(rollout.get("smoke_games_per_branch", 0)) < 100 or not rollout.get("swap_seats"):
        raise ValueError("100-game branch smoke and seat swapping are required")
    if rollout.get("engine_seed_controlled") is not False:
        raise ValueError("engine seed limitation must remain explicit")
    if (policy.get("ppo") or {}).get("algorithm") != "masked_ppo":
        raise ValueError("masked PPO is required")
    if float((policy.get("ppo") or {}).get("kl_coef", 0)) <= 0:
        raise ValueError("positive KL constraint is required")
    if (policy.get("release_gates") or {}).get("submission_replacement_authorized") is not False:
        raise ValueError("handoff package must not authorize submission replacement")
    checked_hash(root, policy["shared_checkpoint"], "path", "sha256")
    for branch in branches:
        checked_hash(root, branch, "deck_path", "deck_sha256")
        checked_hash(root, branch, "adapter_path", "adapter_sha256")

    manifest_path = root / "HANDOFF_MANIFEST.json"
    manifest_files = 0
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "top2_rl_handoff_manifest_v1":
            raise ValueError("unsupported handoff manifest schema")
        for relative, expected in (manifest.get("sha256") or {}).items():
            path = root / relative
            if not path.is_file() or sha256(path).lower() != str(expected).lower():
                raise ValueError(f"manifest payload mismatch: {relative}")
            manifest_files += 1
    return {
        "schema_version": "top2_rl_handoff_verification_v1",
        "root": str(root),
        "roles": roles,
        "deck_ids": [branch["deck_id"] for branch in branches],
        "verified_hashes": 5,
        "manifest_files": manifest_files,
        "reserve_budget_ratio": ratio,
        "holdout_percent": 20,
        "submission_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = verify_policy_root(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
