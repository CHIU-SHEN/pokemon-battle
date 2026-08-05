# MCTS Teacher v2 All-in-One Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one resumable package that collects and freezes 10,000 primary MCTS games on CPU, stops all collectors, then trains and selects a safe teacher checkpoint on a V100.

**Architecture:** Pure planning, auditing, benchmarking, and stage-state logic lives in focused `src/rl` modules. CLI scripts perform process orchestration and artifact I/O, while one Bash job runs the stages sequentially. The frozen dataset archive remains the contract between collection and training even though both stages run on one machine.

**Tech Stack:** Python 3, PyTorch, subprocess, JSON, tarfile, hashlib, pytest, Bash.

## Global Constraints

- Primary branch only; reserve identities are rejected.
- Stages are 10-game smoke, 200-game 12/16/20-worker benchmark, 2,400-game gate, 10,000 total games, frozen dataset, GPU smoke, and GPU training.
- Benchmark games and gate games count toward the 10,000 total after validation.
- Collection and training never overlap; training requires proof that managed collectors stopped.
- `exceptions=0`, `illegal_actions=0`, and `fallback_rate=0` are hard gates.
- Production training requires CUDA and keeps holdout KL at or below `0.03` for `best_safe.pt`.
- Minimum convergence time is 30 minutes; relative-update EMA threshold is `1e-5` with patience 5 by default.
- Every packaged `.sh` file uses LF line endings.

---

### Task 1: Collection planning, auditing, and truthful counters

**Files:**
- Create: `src/rl/mcts_collection.py`
- Modify: `scripts/collect_top2_mcts.py`
- Create: `tests/test_mcts_collection.py`

**Interfaces:**
- Produces: `ShardPlan`, `choose_worker_candidates()`, `plan_shards()`, `audit_collection()`, and cumulative progress reconstruction.

- [ ] Write failing tests covering candidates `(12, 16, 20)`, explicit overrides, disjoint shard IDs/seeds/paths, completed-game exclusion, duplicate IDs, hidden fields, identity mismatches, split presence, and cumulative resume totals.
- [ ] Run `pytest tests/test_mcts_collection.py -q` and confirm missing-interface failures.
- [ ] Implement immutable shard planning and full-file audit helpers; never trust incremental progress as the cumulative source of truth.
- [ ] Record actual action source per decision in `collect_top2_mcts.py`, derive fallback totals from stored records, and atomically rewrite progress.
- [ ] Run the focused test plus existing collector/data tests, then commit `feat: add resumable MCTS collection planning`.

### Task 2: Benchmark selection and sequential stage machine

**Files:**
- Create: `src/rl/mcts_pipeline.py`
- Create: `scripts/run_mcts_all_in_one.py`
- Create: `tests/test_mcts_pipeline.py`

**Interfaces:**
- Consumes: Task 1 planning/audit interfaces.
- Produces: `PipelineState`, `select_worker_count()`, `assert_collectors_stopped()`, atomic `pipeline_state.json`, and dry-run-capable stage commands.

- [ ] Write failing tests showing that the fastest safe candidate under 80% RAM is selected, unsafe candidates are rejected, benchmark games remain scheduled only once, stages cannot skip gates, and training cannot start with a live managed PID.
- [ ] Run `pytest tests/test_mcts_pipeline.py -q` and verify RED.
- [ ] Implement state serialization, legal transitions, throughput selection, PID liveness checks, and command construction with `--device cpu` plus all four thread limits set to `1`.
- [ ] Implement the CLI stages `verify`, `smoke`, `benchmark`, `gate`, `collect`, `freeze`, `gpu-smoke`, `train`, and `all`; each transition re-audits its inputs.
- [ ] Run pipeline and collection tests, exercise `--dry-run all`, then commit `feat: orchestrate single-server MCTS pipeline`.

### Task 3: Frozen dataset archive and stable merge

**Files:**
- Create: `scripts/build_mcts_primary_dataset.py`
- Create: `scripts/verify_mcts_primary_dataset.py`
- Modify: `scripts/train_top2_mcts.py`
- Create: `tests/test_mcts_primary_dataset.py`

**Interfaces:**
- Produces: `mcts-primary-dataset-v2.tar.gz`, manifest schema `mcts_primary_dataset_v2`, per-member SHA-256, and stable game-ID split assignment.

- [ ] Write failing tests for incomplete shards, nonzero safety counters, duplicate IDs, hidden fields, identity mismatch, member hash tampering, stable splits, and deduplicated merging with the authoritative 600-game archive.
- [ ] Verify RED with `pytest tests/test_mcts_primary_dataset.py -q`.
- [ ] Implement deterministic archive creation and verification, including benchmark inventory, search parameters, seeds, provenance, totals, split counts, and hashes.
- [ ] Make the trainer accept only verified archives and deduplicate merged sources by game ID without changing stable splits.
- [ ] Build and verify a fixture archive, run focused tests, then commit `feat: freeze verified MCTS primary datasets`.

### Task 4: Adaptive KL, convergence timing, and safe checkpoints

**Files:**
- Modify: `src/rl/mcts_teacher.py`
- Modify: `scripts/train_top2_mcts.py`
- Modify: `tests/test_mcts_teacher.py`
- Modify: `tests/test_mcts_teacher_train.py`

**Interfaces:**
- Produces: `adapt_kl_coefficient()`, `is_safe_checkpoint()`, `ConvergenceTracker`, `last.pt`, and `best_safe.pt`.

- [ ] Write failing boundary tests for the four KL regions, 1% value-loss tolerance, unsafe latest state, minimum 30-minute convergence time, five-epoch patience, holdout plateau, time-limit reporting, and resume reapplication of learning rate/KL.
- [ ] Run focused tests and verify RED.
- [ ] Implement adaptive KL and apply the new coefficient to the next epoch.
- [ ] Atomically save every latest state to `last.pt`; promote only qualifying epochs to `best_safe.pt`; record the exact terminal stop reason.
- [ ] Implement time-aware convergence and plateau tracking without a fixed eight-epoch stop, and reapply CLI learning rate plus saved adaptive KL after optimizer restore.
- [ ] Run focused tests and CPU train/resume smoke, then commit `feat: select converged safe MCTS teachers`.

### Task 5: Unified handoff, resilient job, and end-to-end verification

**Files:**
- Create: `scripts/build_mcts_all_in_one_handoff.py`
- Create: `scripts/verify_mcts_all_in_one_handoff.py`
- Create: `jobs/mcts_teacher_v2_all_in_one.sh`
- Create: `docs/operations/MCTS_TEACHER_V2_ALL_IN_ONE.md`
- Create: `tests/test_mcts_all_in_one_handoff.py`
- Modify: `.gitattributes`
- Modify: `NEXT_STEPS.md`
- Modify: `项目进度.md`
- Modify: `docs/PROJECT_STRUCTURE.md`

**Interfaces:**
- Produces: `mcts-teacher-v2-all-in-one.tar.gz` and final `mcts-teacher-v2-results.tar.gz`.

- [ ] Write failing tests for exact package members, frozen identities, hashes, LF-only shell members, sequential job order, collector shutdown before GPU smoke, resume invocation, and final result inventory.
- [ ] Run `pytest tests/test_mcts_all_in_one_handoff.py -q` and verify RED.
- [ ] Implement explicit package manifests and verifiers; normalize packaged shell bytes to LF before hashing.
- [ ] Implement the resilient shell job with stage-specific time limits, atomic reports, resume behavior, and environment defaults for the 48-vCPU/V100 target.
- [ ] Document upload, verification, tmux execution, status inspection, resume, expected duration, results download, and Arena handoff; update project tracking docs.
- [ ] Run all focused MCTS tests, then `pytest tests -q`, build and verify the real package, run `git diff --check`, and commit `feat: package all-in-one MCTS teacher pipeline`.
