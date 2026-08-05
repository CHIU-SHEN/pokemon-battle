# MCTS CPU Collector and GPU Trainer Split Implementation Plan (Superseded)

> Superseded by `2026-08-05-mcts-all-in-one-pipeline.md`. Retained for historical traceability; do not execute this plan for the current single-server target.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce independent CPU-only MCTS collection and GPU-only teacher-training packages joined by a checksum-verified dataset archive contract.

**Architecture:** Pure shard planning and dataset auditing live under `src/rl`; CLI scripts orchestrate worker processes, rebuild cumulative summaries, and package immutable artifacts. The teacher trainer retains the existing loss pipeline but adds adaptive KL and separate latest/safe checkpoints. Two package builders declare non-overlapping operational entry points and normalize shell scripts to LF.

**Tech Stack:** Python 3, PyTorch, multiprocessing/subprocess, JSON, tarfile, pytest, Bash.

## Global Constraints

- Primary only; reserve data and checkpoints are rejected.
- CPU stages are 10-game smoke, 2,400-game throughput gate, then 10,000 total games.
- CPU workers write separate shard directories and run with CPU inference plus one BLAS thread.
- `exceptions=0`, `illegal_actions=0`, and `fallback_rate=0` are hard gates.
- GPU training keeps KL hard limit 0.03 and never promotes an unsafe `last.pt`.
- Every packaged `.sh` file uses LF line endings.

---

### Task 1: CPU shard planning and cumulative auditing

**Files:**
- Create: `src/rl/mcts_collection.py`
- Modify: `scripts/collect_top2_mcts.py`
- Create: `tests/test_mcts_collection.py`

**Interfaces:**
- Produces: `choose_worker_count(logical_cpus, override)`, `plan_shards(total_games, workers, seed, iteration_id)`, `audit_collection(root, identity)`.

- [ ] Write failing tests for worker tiers, even shard sizes, unique seeds/IDs/paths, duplicate game rejection, split counts, hidden-field rejection, and cumulative resume totals.
- [ ] Run `pytest tests/test_mcts_collection.py -q` and verify missing-interface failures.
- [ ] Implement pure planning/audit helpers with literal identity checks and full game-file rescans.
- [ ] Add real fallback/action-source counters to each collected game and cumulative progress rebuilds.
- [ ] Run focused tests and preserve existing MCTS dataset tests.

### Task 2: CPU orchestration and dataset archive

**Files:**
- Create: `scripts/run_mcts_cpu_collection.py`
- Create: `scripts/build_mcts_cpu_dataset.py`
- Create: `scripts/verify_mcts_cpu_dataset.py`
- Create: `jobs/mcts_cpu_collector_resilient.sh`
- Create: `tests/test_mcts_cpu_pipeline.py`

**Interfaces:**
- Consumes: Task 1 shard plans/audits.
- Produces: staged collection reports and `mcts-primary-dataset-v2.tar.gz` with `mcts_primary_dataset_v2` manifest.

- [ ] Write failing tests for 10/2,400/10,000 stage gates, subprocess commands forcing CPU and one BLAS thread, incomplete-shard rejection, manifest identity, and member hashes.
- [ ] Implement a dry-run-capable orchestrator whose workers always use separate roots and atomic stage JSON.
- [ ] Implement archive builder/verifier with duplicate IDs, hidden fields, safety totals, split presence, and SHA-256 enforcement.
- [ ] Implement a 24-hour resumable shell job that defaults to smoke and requires explicit flags for 2,400/10,000 stages.
- [ ] Run CPU pipeline tests.

### Task 3: Adaptive KL and best-safe checkpoints

**Files:**
- Modify: `src/rl/mcts_teacher.py`
- Modify: `scripts/train_top2_mcts.py`
- Modify: `tests/test_mcts_teacher.py`
- Modify: `tests/test_mcts_teacher_train.py`

**Interfaces:**
- Produces: `adapt_kl_coefficient(holdout_kl, current, ...)`, `is_safe_checkpoint(metrics, best_metrics, ...)`, `best_safe.pt` and latest `last.pt`.

- [ ] Write failing boundary tests for all four KL regions and safe-checkpoint policy/value comparisons.
- [ ] Verify RED with focused tests.
- [ ] Implement adaptive KL and apply it to the next epoch's loss.
- [ ] Save `best_safe.pt` atomically before unsafe-stop evaluation; keep unsafe latest state only in `last.pt`.
- [ ] On resume, restore optimizer tensors but explicitly reapply CLI learning rate and stored adaptive KL.
- [ ] Run focused teacher tests and real CPU train/resume smoke.

### Task 4: Separate CPU and GPU handoff packages

**Files:**
- Create: `scripts/build_mcts_cpu_collector_handoff.py`
- Create: `scripts/verify_mcts_cpu_collector_handoff.py`
- Create: `scripts/build_mcts_gpu_trainer_handoff.py`
- Create: `scripts/verify_mcts_gpu_trainer_handoff.py`
- Create: `jobs/mcts_gpu_trainer_resilient.sh`
- Create: `docs/operations/MCTS_CPU_COLLECTOR_HANDOFF.md`
- Create: `docs/operations/MCTS_GPU_TRAINER_HANDOFF.md`
- Create: `.gitattributes`
- Create: `tests/test_mcts_split_handoffs.py`

**Interfaces:**
- Produces: `mcts-cpu-collector-v1.tar.gz` and `mcts-gpu-trainer-v2.tar.gz`.

- [ ] Write failing archive-content tests proving the CPU package has no GPU trainer job and the GPU package has no collector job.
- [ ] Assert every packaged shell member contains no carriage-return byte.
- [ ] Implement explicit file manifests, frozen-asset identity, and verifiers.
- [ ] Add `.gitattributes` rule `*.sh text eol=lf` and normalize staged copies before hashing.
- [ ] Run archive tests and build both real packages.

### Task 5: End-to-end verification

**Files:**
- Modify only files required by attributable failures.

- [ ] Run all new focused tests plus existing MCTS tests.
- [ ] Run `pytest tests -q` with the workspace-local temp directory and frozen-root environment.
- [ ] Build and verify both final packages.
- [ ] Run `git diff --check` and confirm a clean feature worktree.
