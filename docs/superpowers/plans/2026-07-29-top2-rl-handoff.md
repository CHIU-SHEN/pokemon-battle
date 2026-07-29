# Top2 RL Handoff Implementation Plan

> **Execution note:** Execute inline in this isolated worktree; no sub-agents were requested.

**Goal:** Build and independently verify a portable Top2 masked-PPO handoff package without starting long training or changing the released submission.

**Architecture:** A branch-bound rollout agent samples only legal mandatory single-choice decisions from the frozen SL-0 + Adapter policy and records behavior statistics. A branch-bound PPO trainer reads only the stable 80% train split, computes GAE/clipped objectives with KL to the frozen initial policy, and writes independent candidate checkpoints. A package builder copies the exact frozen inputs and runtime, creates hashes, and a verifier rejects identity, split, budget, or payload drift.

**Tech Stack:** Python 3.11, PyTorch 2.7, existing `submission/cg` battle engine, JSON/JSONL, tar.gz.

---

### Task 1: Lock the rollout/PPO contracts with failing tests

**Files:**

- Create: `tests/test_top2_rl_handoff.py`
- Create later: `src/rl/top2_ppo.py`
- Create later: `src/rl/top2_rollout.py`
- Create later: `scripts/verify_top2_rl_handoff.py`

1. Add literal tests for stable 80/10/10 game splitting, GAE terminal-return behavior, clipped masked-PPO loss finiteness, and rejection of non-train samples.
2. Add a verifier integration test using a synthetic package root with two distinct branch identities and literal hashes.
3. Run `python tests/test_top2_rl_handoff.py` and confirm the missing modules/commands fail for the expected reason.

### Task 2: Implement the branch-bound RL core

**Files:**

- Create: `src/rl/__init__.py`
- Create: `src/rl/top2_rollout.py`
- Create: `src/rl/top2_ppo.py`
- Create: `src/arena/ppo_agent.py`
- Create: `scripts/collect_top2_rollouts.py`
- Create: `scripts/train_top2_ppo.py`

1. Implement stable game split and GAE with terminal rewards.
2. Implement the branch-bound sampling agent and `top2_rl_rollout_v1` writer.
3. Implement padded minibatch collation, clipped PPO policy/value/entropy/KL losses, train-only enforcement, checkpoint writing, and a PPO Arena loader.
4. Run the focused test until green, then refactor without broadening the action gate.

### Task 3: Freeze configuration and V1/Arena handoff commands

**Files:**

- Create: `config/top2_rl_policy.json`
- Create: `scripts/evaluate_top2_ppo.py`
- Update: `scripts/select_v1_candidates.py`
- Update: `scripts/run_v1_reanalysis.py`
- Create: `TOP2_RL_SERVER_HANDOFF.md`

1. Record exact Top2 identities, smoke budgets, 40% reserve ratio, opponent pool, PPO defaults, and release gates.
2. Add optional branch/deck filters to V1 selection/reanalysis without changing their old defaults.
3. Document Windows and Linux commands from verification through smoke, rollout, V1, PPO smoke, and Arena; explicitly stop before release.
4. Extend tests to ensure branch filters cannot cross deck streams.

### Task 4: Build and validate the portable archive

**Files:**

- Create: `scripts/build_top2_rl_handoff.py`
- Create: `scripts/verify_top2_rl_handoff.py`
- Update: `tests/test_top2_rl_handoff.py`

1. Write a failing integration test for required payload, archive root, manifest, and SHA sidecar.
2. Implement `--source-root` so the isolated worktree can package frozen artifacts from the clean main workspace without copying them into Git.
3. Include only Top2 inputs and runtime dependencies; exclude raw datasets, raw rollout output, `last.pt`, and previous server archives.
4. Build `server_uploads/pokemon-tcg-top2-rl-handoff-v1.tar.gz` and sidecar.
5. Extract into a fresh temporary directory and run the verifier from inside the extracted package.

### Task 5: Update project status and perform fresh verification

**Files:**

- Update: `项目进度.md`
- Create: `reports/top2_rl_handoff_report.json`

1. Mark the RL implementation/handoff layer complete while leaving rollout generation, V1 production, PPO training, and final release incomplete.
2. Run focused RL tests, relevant existing adapter/conversion tests with an explicit `PYTHONPATH`, Python compile checks, package build, sidecar verification, and independent extracted-package verification.
3. Inspect `git diff --check`, `git status --short`, archive size/hash, and report the next single action: run 100 games per branch and decide final compute location from measured throughput.
