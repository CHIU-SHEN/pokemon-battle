# Top2 Gated Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two isolated, resumable masked-PPO self-play loops with best/candidate/history pools, adaptive Arena promotion gates, and server execution packaging.

**Architecture:** Pure state/pool/gate primitives live in focused modules under `src/rl/`; a branch runner composes existing rollout, PPO, holdout, and Arena scripts without allowing checkpoints or data to cross branches. Each stage records an atomic state transition so failed jobs resume safely, while promotion moves the old best into immutable history before changing the best pointer.

**Tech Stack:** Python 3.11, PyTorch, JSON manifests, existing `eval.run_match` engine, PowerShell/Linux shell entry points, SLURM.

## Global Constraints

- Primary and reserve have separate state, data, checkpoint, holdout, and history roots.
- Default rollout budget is 3,000 games per branch per iteration.
- Candidate-vs-best starts at 1,000 games and may extend to at most 3,000.
- At 1,000 games: non-draw win rate `>= 0.58` advances to regression gates; `<= 0.52` rejects; otherwise continue.
- A continued gate requires non-draw win rate `>= 0.55` and Wilson 95% lower bound `> 0.52`.
- Training must reject non-train rows, reused iteration data, deck mismatch, candidate mismatch, or hash mismatch.
- Promotion never modifies `submission/deck.csv` and never authorizes release.
- Formal runs target the server; local runs use reduced smoke budgets.

---

### Task 1: Self-play state and immutable policy pool

**Files:**
- Create: `src/rl/selfplay_state.py`
- Create: `tests/test_selfplay_state.py`

**Interfaces:**
- Produces: `SelfPlayState.load_or_initialize(root: Path, branch: str, deck_id: str, initial_checkpoint: Path) -> SelfPlayState`
- Produces: `SelfPlayState.begin_iteration(iteration_id: str) -> dict`
- Produces: `SelfPlayState.complete_stage(stage: str, artifacts: dict) -> None`
- Produces: `SelfPlayState.promote(candidate: Path, metrics: dict) -> None`
- Produces: `SelfPlayState.reject(reason: str, metrics: dict) -> None`

- [ ] **Step 1: Write failing state tests**

```python
def test_primary_and_reserve_roots_cannot_cross(tmp_path):
    primary = SelfPlayState.load_or_initialize(tmp_path / "primary", "primary", "deck-p", checkpoint(tmp_path, "p"))
    with pytest.raises(ValueError, match="branch"):
        SelfPlayState.load(primary.path, expected_branch="reserve")

def test_promote_archives_old_best_before_switching_pointer(tmp_path):
    state = initialized_state(tmp_path)
    old_sha = state.best["sha256"]
    state.begin_iteration("iter-0001")
    state.promote(checkpoint(tmp_path, "candidate"), {"win_rate": 0.60})
    assert state.history[-1]["sha256"] == old_sha
    assert state.best["sha256"] != old_sha
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_selfplay_state.py -q`  
Expected: FAIL because `src.rl.selfplay_state` does not exist.

- [ ] **Step 3: Implement atomic state persistence**

Implement a dataclass-backed JSON state with schema `top2_selfplay_state_v1`. Write to
`state.json.tmp`, flush, then replace `state.json`. Store relative checkpoint paths and
SHA-256. Validate branch, deck identity, monotonic iteration IDs, and allowed stage order.
Copy promoted checkpoints into content-addressed `best/` and `history/` paths; never mutate
history files.

- [ ] **Step 4: Run state tests**

Run: `python -m pytest tests/test_selfplay_state.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rl/selfplay_state.py tests/test_selfplay_state.py
git commit -m "feat: add isolated self-play state"
```

### Task 2: Opponent pool and iteration-safe rollout manifests

**Files:**
- Create: `src/rl/selfplay_pool.py`
- Modify: `src/rl/top2_rollout.py`
- Modify: `scripts/collect_top2_rollouts.py`
- Create: `tests/test_selfplay_pool.py`
- Modify: `tests/test_top2_rl_handoff.py`

**Interfaces:**
- Consumes: `SelfPlayState.best`, `SelfPlayState.history`
- Produces: `build_opponent_schedule(state: SelfPlayState, games: int, seed: int) -> list[OpponentSpec]`
- Produces: rollout CLI flags `--selfplay-root`, `--iteration-id`, `--learner-checkpoint`, `--schedule-manifest`

- [ ] **Step 1: Write failing pool and manifest tests**

```python
def test_empty_history_weight_falls_back_to_best():
    schedule = build_opponent_schedule(state(history=[]), games=100, seed=7)
    assert count(schedule, "best") == 80
    assert count(schedule, "baseline") == 20

def test_rollout_rejects_checkpoint_from_other_branch(tmp_path):
    with pytest.raises(ValueError, match="deck_id"):
        load_learner_checkpoint(primary_config(), reserve_checkpoint(tmp_path))

def test_iteration_manifest_records_both_policy_hashes(tmp_path):
    manifest = collect_fixture_iteration(tmp_path)
    assert all(row["learner_sha256"] and row["opponent_sha256"] for row in manifest["games"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_selfplay_pool.py tests/test_top2_rl_handoff.py -q`  
Expected: new pool tests FAIL.

- [ ] **Step 3: Implement pool and checkpoint-aware rollout**

Represent each opponent as `{kind, checkpoint, checkpoint_sha256, weight, branch, deck_id}`.
Use deterministic weighted scheduling with exact 50/30/20 quotas; when history is empty,
assign its quota to best. Extend `Top2RolloutAgent` checkpoint loading to expose the loaded
hash. Extend the collector to load the learner from current best, instantiate best/history
opponents with `Top2RolloutAgent(record_decisions=False)`, and write iteration and both
policy hashes into every game document.

- [ ] **Step 4: Run pool and existing handoff tests**

Run: `python -m pytest tests/test_selfplay_pool.py tests/test_top2_rl_handoff.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rl/selfplay_pool.py src/rl/top2_rollout.py scripts/collect_top2_rollouts.py tests/test_selfplay_pool.py tests/test_top2_rl_handoff.py
git commit -m "feat: collect policy-pool self-play rollouts"
```

### Task 3: Adaptive Arena promotion gate

**Files:**
- Create: `src/rl/selfplay_gate.py`
- Create: `scripts/evaluate_selfplay_gate.py`
- Create: `tests/test_selfplay_gate.py`

**Interfaces:**
- Produces: `wilson_interval(wins: int, losses: int, z: float = 1.959963984540054) -> tuple[float, float]`
- Produces: `gate_decision(wins: int, losses: int, draws: int, games_cap: int = 3000) -> GateDecision`
- Produces: CLI exit codes `0=promote-ready`, `2=reject`, `3=continue`

- [ ] **Step 1: Write failing boundary tests**

```python
@pytest.mark.parametrize(
    "wins,losses,draws,expected",
    [(580, 420, 0, "promote_ready"), (520, 480, 0, "reject"), (550, 450, 0, "continue")],
)
def test_initial_thousand_game_gate(wins, losses, draws, expected):
    assert gate_decision(wins, losses, draws).status == expected

def test_final_gate_requires_point_and_wilson_thresholds():
    assert gate_decision(1680, 1320, 0).status == "promote_ready"
    assert gate_decision(1590, 1410, 0).status == "reject"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_selfplay_gate.py -q`  
Expected: FAIL because gate module does not exist.

- [ ] **Step 3: Implement gate math and report CLI**

Ignore draws in the win-rate denominator but retain them in reports. At exactly 1,000
completed games apply the 58/52 direct thresholds. Between 1,001 and 2,999 games return
`continue` unless statistical rejection is already certain. At 3,000 games require point
estimate `>= 0.55` and Wilson lower bound `> 0.52`. Emit JSON containing counts, point
estimate, interval, threshold reason, next game target, and status.

- [ ] **Step 4: Run gate tests**

Run: `python -m pytest tests/test_selfplay_gate.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rl/selfplay_gate.py scripts/evaluate_selfplay_gate.py tests/test_selfplay_gate.py
git commit -m "feat: add adaptive self-play promotion gate"
```

### Task 4: Resumable branch iteration runner

**Files:**
- Create: `src/rl/selfplay_runner.py`
- Create: `scripts/run_top2_selfplay_iteration.py`
- Create: `tests/test_selfplay_runner.py`
- Modify: `scripts/train_top2_ppo.py`

**Interfaces:**
- Consumes: state, opponent schedule, existing PPO and evaluation scripts, `GateDecision`
- Produces: `run_iteration(config: IterationConfig) -> IterationReport`
- Produces: CLI `--branch`, `--selfplay-root`, `--iteration-id`, `--rollout-games`, `--gate-games`, `--gate-cap`, `--resume`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_resume_skips_completed_rollout(tmp_path, fake_stages):
    runner = configured_runner(tmp_path, fake_stages)
    runner.run(stop_after="train")
    runner.run(resume=True)
    assert fake_stages.calls["rollout"] == 1

def test_rejected_candidate_does_not_change_best(tmp_path, fake_stages):
    runner = configured_runner(tmp_path, fake_stages, gate="reject")
    before = runner.state.best["sha256"]
    runner.run()
    assert runner.state.best["sha256"] == before
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_selfplay_runner.py -q`  
Expected: FAIL because runner does not exist.

- [ ] **Step 3: Implement stage orchestration**

Use injectable stage callables for tests. Production stages collect fresh rollout, validate
the manifest iteration ID, train from current best, run holdout, execute Arena in 1,000-game
batches, aggregate gate reports, run regression matrices, and call promote/reject. Update
state only after verifying each stage artifact and hash. Add `--initial-checkpoint` to PPO
training so a new candidate always starts from current best instead of the frozen original.

- [ ] **Step 4: Run runner and RL tests**

Run: `python -m pytest tests/test_selfplay_runner.py tests/test_top2_rl_handoff.py tests/test_selfplay_gate.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rl/selfplay_runner.py scripts/run_top2_selfplay_iteration.py scripts/train_top2_ppo.py tests/test_selfplay_runner.py
git commit -m "feat: orchestrate resumable self-play iterations"
```

### Task 5: Server jobs, handoff, documentation, and end-to-end smoke

**Files:**
- Create: `jobs/top2_selfplay_rollout.slurm`
- Create: `jobs/top2_selfplay_train.slurm`
- Create: `jobs/top2_selfplay_gate.slurm`
- Create: `scripts/build_top2_selfplay_handoff.py`
- Create: `scripts/verify_top2_selfplay_handoff.py`
- Create: `tests/test_top2_selfplay_handoff.py`
- Modify: `项目进度.md`
- Modify: `reports/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: branch iteration CLI and self-play config
- Produces: verified server archive with source, frozen inputs, jobs, manifest, and hashes

- [ ] **Step 1: Write failing handoff test**

```python
def test_handoff_contains_parallel_branch_jobs_and_no_training_data(tmp_path):
    package = build_fixture_handoff(tmp_path)
    assert package.manifest["branches"] == ["primary", "reserve"]
    assert package.manifest["default_iterations"] == 1
    assert not any(path.suffix == ".jsonl" for path in package.files)
```

- [ ] **Step 2: Run test and verify failure**

Run: `python -m pytest tests/test_top2_selfplay_handoff.py -q`  
Expected: FAIL because handoff builder does not exist.

- [ ] **Step 3: Implement jobs and verified package**

Build separate rollout jobs for primary/reserve, GPU array training, and batched gate jobs.
Default the first server submission to one iteration; provide an explicit follow-up command
for four more iterations after result review. Package source, configs, frozen checkpoint
hashes, job files, tests, and a manifest, excluding training JSONL, generated trajectories,
logs, and returned result archives.

- [ ] **Step 4: Correct progress documentation**

Record the prior server run as a one-shot PPO pilot, not completed self-play. State that the
new loop is implemented but formal five-round server execution remains pending. Preserve the
existing result report as pilot evidence.

- [ ] **Step 5: Run end-to-end reduced smoke**

Run:

```powershell
$env:PYTHONPATH='.'
python scripts/run_top2_selfplay_iteration.py --branch primary --selfplay-root artifacts/dev_smoke/selfplay --iteration-id smoke-0001 --rollout-games 10 --gate-games 20 --gate-cap 20
python scripts/verify_top2_selfplay_handoff.py
python -m pytest tests/test_selfplay_state.py tests/test_selfplay_pool.py tests/test_selfplay_gate.py tests/test_selfplay_runner.py tests/test_top2_selfplay_handoff.py -q
```

Expected: smoke writes a completed or rejected iteration without changing submission;
handoff verification succeeds; all listed tests PASS.

- [ ] **Step 6: Commit**

```bash
git add jobs scripts tests .gitignore 项目进度.md reports/README.md
git commit -m "build: prepare gated Top2 self-play handoff"
```
