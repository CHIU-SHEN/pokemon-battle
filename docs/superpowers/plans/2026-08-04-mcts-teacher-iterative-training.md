# MCTS Teacher Iterative Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current one-shot MCTS adapter trainer into a resumable primary teacher trainer with policy/value optimization, convergence diagnostics, holdout gates, and wall-time-safe checkpoints.

**Architecture:** Pure convergence and parameter-statistics helpers live in `src/rl/mcts_teacher.py`; the existing MCTS loader/loss remains responsible for joint-action visit targets. `scripts/train_top2_mcts.py` becomes the executable training boundary, while a dedicated smoke runner and v2 handoff builder verify the complete server workflow without starting long jobs.

**Tech Stack:** Python 3, PyTorch, pytest, JSON checkpoints, tarfile.

## Global Constraints

- Process primary first; primary and reserve data/checkpoints must never mix.
- Freeze the shared base trunk; train adapter, `policy_delta`, and `value_delta`.
- Platform convergence requires 3 consecutive windows with relative-update EMA `< 1e-5` and holdout policy improvement `< 0.2%`.
- Holdout value loss may not worsen by more than `1.0%`.
- Local smoke limit is 1,800 seconds, one server train limit is 21,600 seconds, and one full iteration limit is 86,400 seconds.
- `exceptions=0`, `illegal_actions=0`, and `fallback_rate=0` are hard gates.
- Never serialize hidden opponent or belief-particle fields.

---

### Task 1: Convergence and parameter-update primitives

**Files:**
- Create: `src/rl/mcts_teacher.py`
- Create: `tests/test_mcts_teacher.py`

**Interfaces:**
- Produces: `TeacherConvergenceConfig`, `gradient_norm(parameters)`, `snapshot_parameters(parameters)`, `relative_parameter_update(before, parameters)`, and `evaluate_teacher_stop(history, config)`.

- [ ] **Step 1: Write failing hand-calculated statistic tests**

```python
def test_relative_parameter_update_matches_hand_calculation():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    before = snapshot_parameters([parameter])
    parameter.data.add_(torch.tensor([0.3, 0.4]))
    assert relative_parameter_update(before, [parameter]) == pytest.approx(0.1)
```

- [ ] **Step 2: Run the new test and verify import failure**

Run: `pytest tests/test_mcts_teacher.py -q`
Expected: FAIL because `src.rl.mcts_teacher` does not exist.

- [ ] **Step 3: Implement finite statistic helpers and EMA-based history fields**

Use flattened L2 norms without concatenating tensors; reject mismatched snapshots and non-finite values.

- [ ] **Step 4: Add platform and false-convergence tests**

```python
def test_small_updates_do_not_stop_while_holdout_improves():
    history = [_window(1e-6, policy_loss=value) for value in (1.0, 0.9, 0.8)]
    assert evaluate_teacher_stop(history, TeacherConvergenceConfig()).stop is False
```

Also test three stable windows, two worsening policy windows, KL violation, non-finite metrics, and wall-time stop being distinct from convergence.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_mcts_teacher.py -q`
Expected: PASS.

### Task 2: Holdout evaluation and resumable teacher checkpoint

**Files:**
- Modify: `src/rl/mcts_train.py`
- Modify: `scripts/train_top2_mcts.py`
- Modify: `tests/test_mcts_train.py`
- Create: `tests/test_mcts_teacher_train.py`

**Interfaces:**
- Consumes: Task 1 convergence/statistic helpers.
- Produces: `evaluate_mcts_rows(model, reference, rows, ...) -> dict[str, float]` and checkpoint schema `top2_mcts_teacher_checkpoint_v2`.

- [ ] **Step 1: Write failing tests for split loading and branch identity**

Exercise real JSON game files with train and valid rows. Assert that a checkpoint whose branch or deck differs is rejected before model state is loaded.

- [ ] **Step 2: Run focused tests and verify expected failures**

Run: `pytest tests/test_mcts_train.py tests/test_mcts_teacher_train.py -q`
Expected: FAIL because v2 evaluation/resume interfaces are absent.

- [ ] **Step 3: Implement no-gradient holdout evaluation**

Compute the same policy/value/KL/entropy metrics as training under `model.eval()` and `torch.no_grad()`. Keep soft visit targets and variable joint actions unchanged.

- [ ] **Step 4: Implement explicit teacher parameter selection**

Set every model parameter frozen, then enable only `adapter`, `policy_delta`, and `value_delta`. Assert the resulting parameter list is non-empty and record trainable parameter names in the summary.

- [ ] **Step 5: Implement step statistics and epoch windows**

Before `optimizer.step()`, record raw gradient norm; apply clipping; record clipped norm; snapshot parameters and calculate relative update after the step. Store epoch averages and EMA values.

- [ ] **Step 6: Implement atomic v2 checkpoint/resume**

Save model state, optimizer state, epoch, Python/Torch RNG states, elapsed time, convergence history, identity hashes, and effective configuration to a temporary path then replace `last.pt`. `--resume` must restore all of them and be mutually exclusive with `--initial-checkpoint`.

- [ ] **Step 7: Add time and convergence termination**

Add `--max-wall-seconds` defaulting to 21,600, `--checkpoint-interval-seconds` defaulting to 1,800, and thresholds matching Global Constraints. Finish the current batch before stopping; emit `time_limit_reached=true` separately from `converged=true`.

- [ ] **Step 8: Run focused training tests**

Run: `pytest tests/test_mcts_train.py tests/test_mcts_teacher.py tests/test_mcts_teacher_train.py -q`
Expected: PASS.

### Task 3: CPU end-to-end smoke

**Files:**
- Create: `scripts/run_mcts_teacher_smoke.py`
- Create: `tests/test_mcts_teacher_smoke.py`

**Interfaces:**
- Consumes: `scripts/train_top2_mcts.py` v2 CLI.
- Produces: `mcts_teacher_smoke_v1` report with checkpoint, summary, resume result, elapsed time, and safety fields.

- [ ] **Step 1: Write a failing smoke integration test**

Build two tiny `top2_mcts_game_v1` fixtures with train/valid splits and run the smoke entry point on CPU for one batch. Assert finite metrics, non-empty trainable names, a v2 checkpoint, and a second invocation that resumes it.

- [ ] **Step 2: Verify the integration test fails because the runner is absent**

Run: `pytest tests/test_mcts_teacher_smoke.py -q`
Expected: FAIL on missing script/module.

- [ ] **Step 3: Implement bounded smoke orchestration**

The runner invokes training with `--device cpu --epochs 1 --max-batches 1 --max-wall-seconds 1800`, verifies output schemas, then performs a resume validation without starting Arena or server work.

- [ ] **Step 4: Run smoke tests**

Run: `pytest tests/test_mcts_teacher_smoke.py -q`
Expected: PASS within the local time limit.

### Task 4: Teacher v2 server handoff

**Files:**
- Create: `scripts/build_mcts_teacher_v2_handoff.py`
- Create: `scripts/verify_mcts_teacher_v2_handoff.py`
- Create: `jobs/mcts_teacher_v2_resilient.sh`
- Create: `docs/operations/MCTS_TEACHER_V2_SERVER_HANDOFF.md`
- Create: `tests/test_mcts_teacher_v2_handoff.py`

**Interfaces:**
- Consumes: teacher trainer, smoke runner, frozen primary assets, and the authoritative complete-results archive.
- Produces: `server_uploads/mcts-distill-v2-teacher.tar.gz`, `.sha256`, and `mcts_teacher_v2_handoff_v1` manifest.

- [ ] **Step 1: Write failing package behavior tests**

Build into `tmp_path`; inspect the tar archive and assert the trainer, convergence helper, smoke runner, resilient job, operations guide, primary frozen assets, authoritative archive hash, and per-file hashes are present. Assert `submission_replacement_authorized` is false.

- [ ] **Step 2: Run the package test and verify failure**

Run: `pytest tests/test_mcts_teacher_v2_handoff.py -q`
Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement builder and verifier**

Use `tarfile`, copy only declared files, reject missing frozen inputs, write hashes for every member, and verify archive checksum plus manifest identity without extracting outside a temporary directory.

- [ ] **Step 4: Implement resilient primary-first job and guide**

The job verifies the package, runs CPU smoke, resumes primary teacher training with 6-hour train and 24-hour iteration caps, and stops before any 100/400-game Arena unless the user starts that stage explicitly.

- [ ] **Step 5: Run package tests and build the real handoff**

Run: `pytest tests/test_mcts_teacher_v2_handoff.py -q`
Expected: PASS.

Run: `python scripts/build_mcts_teacher_v2_handoff.py --output-dir server_uploads`
Expected: archive, checksum, and manifest with no missing inputs.

### Task 5: Full verification

**Files:**
- Modify only files identified by failures attributable to this feature.

- [ ] **Step 1: Run MCTS-focused tests**

Run: `pytest tests/test_mcts_dataset.py tests/test_mcts_train.py tests/test_mcts_teacher.py tests/test_mcts_teacher_train.py tests/test_mcts_teacher_smoke.py tests/test_mcts_teacher_v2_handoff.py -q`
Expected: PASS.

- [ ] **Step 2: Run the complete project suite**

Run: `pytest tests -q`
Expected: PASS with no warnings introduced by this change.

- [ ] **Step 3: Verify repository and artifact integrity**

Run: `git diff --check`
Expected: no output.

Run: `python scripts/verify_mcts_teacher_v2_handoff.py server_uploads/mcts-distill-v2-teacher.tar.gz`
Expected: exit code 0 and authoritative archive hash match.
