# Top2 Local PPO Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a time-bounded local RTX 5060 Top2 pilot, select a preliminary primary PPO configuration from three controlled trials, and build a v2 server handoff containing the evidence and selected starting parameters.

**Architecture:** Pure pilot policy helpers decide budget tiers and recommendations from literal reports. Existing rollout and PPO commands gain explicit override and safety-gate interfaces; a holdout evaluator measures candidate drift on valid/test without exposing those rows to the optimizer. A single orchestrator runs benchmark, two 100-game rollouts, three primary trials, holdout/Arena evaluation, selection, and evidence generation before the package builder creates a v2 archive.

**Tech Stack:** Python 3.11, PyTorch 2.7 CUDA 12.8, existing `submission/cg` battle engine, JSON/JSONL, tar.gz, PowerShell host.

## Global Constraints

- Local wall-clock budget is 7,200 seconds and must never weaken split or safety gates.
- Primary and reserve keep distinct frozen `deck_id` streams; valid/test total 20% and never reach the optimizer.
- Three primary trials start from the same frozen SL-0 + primary Adapter and the same train rollout.
- Hard gates: zero exceptions, zero illegal actions, finite losses, reference KL at most `0.03`, clip fraction at most `0.30`, entropy drop at most 50%.
- Full tier uses 100 rollout games per branch and 200 swapped-seat Arena games per primary trial.
- The result is preliminary server guidance, not a release decision and not authorization to modify `submission/deck.csv`.
- Long server rollout, V1 expansion, full PPO, and final release Arena remain outside this plan.

---

### Task 1: Pilot policy and deterministic selection

**Files:**
- Create: `src/rl/pilot.py`
- Modify: `tests/test_top2_rl_handoff.py`

**Interfaces:**
- Produces: `choose_budget_tier(predicted_seconds: float) -> PilotBudget`
- Produces: `wilson_interval(wins: int, games: int) -> tuple[float, float]`
- Produces: `select_preliminary_trial(trials: list[dict]) -> dict`

- [ ] **Step 1: Write failing tests** for thresholds `<=5400 => full`, `<=7200 => reduced`, `>7200 => minimal`; literal Wilson interval bounds; rejection of unsafe trials; and conservative tie-break when Arena intervals overlap.
- [ ] **Step 2: Run** `python tests/test_top2_rl_handoff.py` and verify imports fail because `src.rl.pilot` does not exist.
- [ ] **Step 3: Implement** immutable `PilotBudget` presets: full `(100, 200, 3 trials)`, reduced `(100, 100, 3 trials)`, minimal `(100, 0, conservative+baseline with 2 epochs)`; implement Wilson 95%; filter trials on `eligible`; sort separated results by lower bound, otherwise choose conservative then baseline.
- [ ] **Step 4: Run** `python tests/test_top2_rl_handoff.py` and confirm all contracts pass.
- [ ] **Step 5: Commit** `src/rl/pilot.py` and the focused tests with message `feat: add Top2 pilot selection policy`.

### Task 2: PPO overrides, safety gates, and run summaries

**Files:**
- Modify: `scripts/train_top2_ppo.py`
- Modify: `config/top2_rl_policy.json`
- Modify: `tests/test_top2_rl_handoff.py`

**Interfaces:**
- Consumes: rollout rows accepted by `load_rollout_rows(path, deck_id)`.
- Produces: CLI overrides `--clip-ratio`, `--kl-coef`, `--entropy-coef`, `--target-kl-max`, `--clip-fraction-max`, `--entropy-drop-max`, `--max-wall-seconds`.
- Produces: `<output>/summary.json` with `status`, `eligible`, `stop_reason`, effective parameters, epoch metrics, wall time, and checkpoint path.

- [ ] **Step 1: Write a failing unit test** for `safety_stop_reason(metrics, first_entropy, limits)` returning `non_finite`, `kl_limit`, `clip_fraction_limit`, `entropy_collapse`, or `None` using literal metrics.
- [ ] **Step 2: Run** the focused test and confirm the helper import fails.
- [ ] **Step 3: Implement** the helper in `src/rl/pilot.py`, expose CLI overrides, apply effective values instead of fixed config values, check gates after each epoch, stop after the current epoch, and always write `summary.json`.
- [ ] **Step 4: Add the three literal presets** to `config/top2_rl_policy.json` and validate their names and values in `scripts/verify_top2_rl_handoff.py`.
- [ ] **Step 5: Run** focused tests plus a one-batch CUDA smoke against the existing development rollout; require `summary.json`, finite metrics, and `eligible=true`.
- [ ] **Step 6: Commit** the trainer, config, verifier, helper, and tests with message `feat: gate Top2 PPO pilot updates`.

### Task 3: Frozen holdout evaluator

**Files:**
- Create: `scripts/evaluate_top2_ppo_holdout.py`
- Modify: `src/rl/top2_ppo.py`
- Modify: `tests/test_top2_rl_handoff.py`

**Interfaces:**
- Produces: `load_rollout_rows_for_splits(path: Path, deck_id: str, allowed_splits: set[str]) -> list[dict]`; it rejects train rows when evaluating holdout.
- Produces: report schema `top2_ppo_holdout_v1` with sample/game counts, candidate/reference action accuracy, action agreement, reference KL, candidate/reference value MSE, illegal argmax, and finite status.

- [ ] **Step 1: Write failing tests** proving the holdout loader accepts only valid/test from one deck and rejects train or cross-deck rows.
- [ ] **Step 2: Run** the focused test and confirm the new loader is absent.
- [ ] **Step 3: Implement** split-aware loading without weakening `load_rollout_rows` train-only behavior.
- [ ] **Step 4: Implement** batch evaluation using the candidate checkpoint and frozen initial Adapter; mask illegal options before argmax and KL.
- [ ] **Step 5: Run** on the existing development rollout if it contains holdout; otherwise generate games until at least one valid/test game exists, then require finite report and zero illegal argmax.
- [ ] **Step 6: Commit** with message `feat: evaluate Top2 PPO frozen holdout`.

### Task 4: Time-bounded local pilot orchestrator

**Files:**
- Create: `scripts/run_top2_local_pilot.py`
- Modify: `tests/test_top2_rl_handoff.py`
- Modify: `TOP2_RL_SERVER_HANDOFF.md`

**Interfaces:**
- Consumes: policy presets, rollout/train/holdout/Arena scripts, `choose_budget_tier`, and `select_preliminary_trial`.
- Produces: `reports/top2_local_pilot_report.json`, `.md`, and `config/top2_rl_selected.json`.

- [ ] **Step 1: Write a failing command-construction test** that verifies all three trials start from the same project root and rollout directory while receiving distinct literal hyperparameters.
- [ ] **Step 2: Run** the focused test and confirm the orchestrator helper is absent.
- [ ] **Step 3: Implement** `build_trial_commands(...)` as a pure helper and a subprocess runner that fails closed on nonzero safety stages.
- [ ] **Step 4: Implement** 10-game benchmark timing, budget-tier prediction, primary/reserve rollout, per-trial PPO, holdout and swapped Arena, elapsed-time checks between atomic stages, and preliminary selection.
- [ ] **Step 5: Write** machine-readable and Markdown reports plus selected config with exact input hashes, hardware/CUDA metadata, stage durations, data counts, trial evidence, recommendation status, and remaining server work.
- [ ] **Step 6: Update** the handoff guide with one local command and one server continuation command.
- [ ] **Step 7: Run** `--help`, pure tests, and a development mode using 2 rollout/Arena games and one PPO batch.
- [ ] **Step 8: Commit** with message `feat: orchestrate Top2 local PPO pilot`.

### Task 5: Execute the RTX 5060 pilot

**Files generated by scripts:**
- Generate ignored runtime data under `experiments/adapter_top2_rl_pilot/<run_id>/`
- Generate tracked `reports/top2_local_pilot_report.json`
- Generate tracked `reports/top2_local_pilot_report.md`
- Generate tracked `config/top2_rl_selected.json`

**Interfaces:**
- Consumes: `python scripts/run_top2_local_pilot.py --project-root <main-root> --device cuda --max-wall-seconds 7200`.
- Produces: a preliminary selected preset or an explicit no-recommendation report.

- [ ] **Step 1: Verify CUDA** with `torch.cuda.is_available()`, device name, total VRAM, and a small tensor operation.
- [ ] **Step 2: Run** the orchestrator with a 7,200-second limit and send commentary updates at stage boundaries.
- [ ] **Step 3: Inspect** both branch summaries for exactly 100 games, zero exceptions/illegal actions, distinct deck IDs, and nonempty train plus holdout splits.
- [ ] **Step 4: Inspect** all attempted trial summaries, holdout reports, and Arena reports; do not override automatic gates.
- [ ] **Step 5: Cross-check** selected config against the report and frozen hashes; record actual elapsed time and peak CUDA memory.
- [ ] **Step 6: Commit** only compact reports/config/code, never raw rollout JSON or temporary checkpoints, with message `results: record Top2 local PPO pilot`.

### Task 6: Build and independently verify the v2 server package

**Files:**
- Modify: `scripts/build_top2_rl_handoff.py`
- Modify: `scripts/verify_top2_rl_handoff.py`
- Modify: `项目进度.md`
- Modify: `reports/top2_rl_handoff_report.json`
- Modify: `tests/test_top2_rl_handoff.py`

**Interfaces:**
- Produces: `server_uploads/pokemon-tcg-top2-rl-handoff-v2.tar.gz` and `.sha256`.

- [ ] **Step 1: Write a failing archive-contract test** expecting v2 to contain the selected config, local JSON/Markdown reports, holdout evaluator, orchestrator, and updated server guide while excluding raw rollout and pilot checkpoints.
- [ ] **Step 2: Update** package basename, required payload, manifest purpose/exclusions, and verifier requirements.
- [ ] **Step 3: Update** project progress to distinguish local pilot completion from remaining server formal training.
- [ ] **Step 4: Build** v2 using edited code from the worktree and frozen artifacts from the main root.
- [ ] **Step 5: Verify** sidecar, extract to a fresh validated temporary directory, run package verifier, contract tests, Top2 inference smoke, Python compile, 2-game rollout, 1-batch PPO, holdout evaluation, 2-game Arena, and 1-item V1 reanalysis from inside the archive.
- [ ] **Step 6: Run** `git diff --check`, inspect both worktree and main status, and report the exact v2 path, size, SHA-256, local elapsed time, preliminary preset, and remaining server command.
- [ ] **Step 7: Commit** code/docs/report changes with message `build: prepare Top2 RL server handoff v2`.
