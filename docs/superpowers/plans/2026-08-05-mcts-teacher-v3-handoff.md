# MCTS Teacher v3 Quality-Gated Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a checksum-verified, resumable server handoff that gates a 5,000-game high-quality MCTS distillation run on a 400-game direct-teacher evaluation.

**Architecture:** Add a small pure teacher-gate module, then compose existing evaluation, collection, audit, dataset, and training scripts in a v3 shell job. A dedicated builder freezes required files into one archive whose verifier checks manifest hashes and whose runbook documents server operation and output retrieval.

**Tech Stack:** Python 3.11+, pytest, Bash, PyTorch, JSON manifests, tar/gzip, SHA-256

## Global Constraints

- Direct teacher gate: 400 swapped-seat games against the deterministic current baseline.
- Teacher passes only with decisive-game win rate at least 0.58 and zero exceptions, illegal actions, and MCTS safety fallbacks.
- Collection defaults: 5,000 games, 128 simulations, 3 particles, maximum depth 10, 16 CPU workers.
- All expensive stages must be resumable and reject incompatible saved identities.
- The package must never automatically promote a checkpoint.
- Shell files inside the archive must use LF line endings.

---

### Task 1: Teacher quality gate

**Files:**
- Create: `src/rl/mcts_teacher_gate.py`
- Create: `scripts/gate_mcts_teacher.py`
- Create: `tests/test_mcts_teacher_gate.py`

**Interfaces:**
- Consumes: evaluation report schema `top2_mcts_eval_v1` produced by `scripts/evaluate_top2_mcts.py`.
- Produces: `teacher_gate_decision(report: Mapping[str, Any], *, minimum_games: int = 400, minimum_win_rate: float = 0.58) -> dict[str, Any]` and a CLI that writes a JSON decision report and exits 0 only on pass.

- [ ] **Step 1: Write failing gate tests**

Test literal reports for: 232/168 passing at exactly 58%; 231/169 failing as weak; any exception, illegal action, or nonzero fallback rate failing as unsafe; fewer than 400 games failing as incomplete. Run `python -m pytest tests/test_mcts_teacher_gate.py -q` and confirm import failure.

- [ ] **Step 2: Implement the pure decision function**

Compute games from wins/losses/draws, decisive win rate from wins/(wins+losses), validate nonnegative counts, and return schema `mcts_teacher_quality_gate_v1` with `status`, `reason`, thresholds, and observed metrics.

- [ ] **Step 3: Implement and test the CLI boundary**

Add arguments `REPORT`, `--output`, `--minimum-games`, and `--minimum-win-rate`. The CLI reads the real report, atomically writes the decision, prints compact JSON, and exits 0 on pass or 2 otherwise. Extend tests to call `main([...])` against temporary files, then run `python -m pytest tests/test_mcts_teacher_gate.py -q`.

### Task 2: Resumable v3 server pipeline

**Files:**
- Create: `jobs/mcts_teacher_v3_quality_gated.sh`
- Create: `tests/test_mcts_teacher_v3_job.py`

**Interfaces:**
- Consumes: `evaluate_top2_mcts.py`, `gate_mcts_teacher.py`, `collect_top2_mcts.py`, dataset/audit scripts, and `train_top2_mcts.py`.
- Produces: run root containing `teacher-eval.json`, `teacher-gate.json`, `collection/audit-5000.json`, frozen dataset, training checkpoints, and final result archive.

- [ ] **Step 1: Write a failing executable job test**

Run the job with `DRY_RUN=1`, a temporary `RUN_ROOT`, and environment overrides. Assert the emitted plan orders smoke, teacher evaluation, gate, collection, audit, dataset verification, training, and result packaging; assert the collection command carries literal defaults 128/3/10 and `--resume`; assert no promotion command exists. Confirm failure because the job is absent.

- [ ] **Step 2: Implement resumable stage markers and commands**

Create the Bash job with `set -euo pipefail`, explicit environment defaults, per-worker shard allocation, managed PID checks, atomic stage completion markers, `--resume` for evaluation/collection/training, and a dry-run function that prints commands without executing them.

- [ ] **Step 3: Implement teacher stop and final result packaging**

Run the gate immediately after evaluation and exit while preserving reports if it fails. On success, finish collection/training and create `mcts-teacher-v3-results.tar.gz` plus SHA-256 containing gate, audit, dataset checksum, summary, best-safe, and last checkpoints. Run `python -m pytest tests/test_mcts_teacher_v3_job.py -q`.

### Task 3: Checksum-verified handoff builder

**Files:**
- Create: `scripts/build_mcts_teacher_v3_handoff.py`
- Create: `scripts/verify_mcts_teacher_v3_handoff.py`
- Create: `docs/MCTS_TEACHER_V3_SERVER_RUNBOOK.md`
- Create: `tests/test_mcts_teacher_v3_handoff.py`

**Interfaces:**
- Consumes: repository code plus the same frozen base/adapter/authoritative artifacts used by v2.
- Produces: `server_uploads/mcts-teacher-v3-quality-gated.tar.gz`, adjacent `.sha256`, and manifest schema `mcts_teacher_v3_quality_gated_v1`.

- [ ] **Step 1: Write the failing archive contract test**

Build into a temporary directory and assert archive/checksum names, complete frozen inputs, manifest defaults 400/0.58/5000/128/3/10/16, verifier success, LF shell content, runbook presence, and all required scripts. Confirm failure because builders do not exist.

- [ ] **Step 2: Implement builder and verifier**

Follow the v2 copy/hash patterns, exclude caches, normalize shell line endings, verify the authoritative archive hash, hash every manifest member, validate the outer checksum and every member in the verifier, and reject missing or unexpected schema data.

- [ ] **Step 3: Write the server runbook**

Document upload, checksum, extraction, verification, smoke, full run, tmux monitoring, worker override, resume behavior, teacher-gate stop interpretation, output download, and expected V100/48-vCPU runtime ranges.

- [ ] **Step 4: Run component and regression tests**

Run `python -m pytest tests/test_mcts_teacher_gate.py tests/test_mcts_teacher_v3_job.py tests/test_mcts_teacher_v3_handoff.py tests/test_mcts_collection.py tests/test_mcts_eval_resume.py -q` and fix only failures caused by this change.

### Task 4: Build and independently verify the deliverable

**Files:**
- Generate: `server_uploads/mcts-teacher-v3-quality-gated.tar.gz`
- Generate: `server_uploads/mcts-teacher-v3-quality-gated.tar.gz.sha256`

**Interfaces:**
- Consumes: completed builder and frozen repository inputs.
- Produces: final upload artifacts and verification evidence.

- [ ] **Step 1: Build the archive**

Run `python scripts/build_mcts_teacher_v3_handoff.py --output-dir server_uploads --frozen-root .` and require an empty `missing_frozen_files` list.

- [ ] **Step 2: Verify from the finished archive**

Run `python scripts/verify_mcts_teacher_v3_handoff.py server_uploads/mcts-teacher-v3-quality-gated.tar.gz` and require `verified: true` with schema `mcts_teacher_v3_quality_gated_v1`.

- [ ] **Step 3: Run the complete relevant test set and inspect status**

Run the component/regression command from Task 3, `git diff --check`, and `git status --short`. Report archive paths, exact SHA-256, test counts, and any unrelated pre-existing files without modifying them.
