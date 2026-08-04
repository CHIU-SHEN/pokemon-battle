# Primary Budgeted MCTS Submission Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an isolated primary Crustle submission candidate with deadline-aware belief-PUCT, complete evaluation gates, and a checksum-verified package.

**Architecture:** Add monotonic deadline and per-game budget controls to the existing belief-PUCT agent, then expose them through a self-contained candidate entry point. Extend evaluation to report latency and budget sources, and build only the primary frozen assets into an isolated archive.

**Tech Stack:** Python 3.10+, PyTorch, pytest, official Search API bindings, JSON, tar/SHA256.

## Global Constraints

- Do not modify or overwrite `submission/`.
- Use `crustle_kangaskhan_cage`, frozen SL-0 shared, and frozen primary Adapter only.
- Defaults are 8 simulations, 1 particle, depth 4, 30ms decision budget.
- Every timeout, exception, or illegal result falls back to frozen primary best.
- Eligibility requires 0 exceptions, 0 illegal actions, p95 at most 35ms, and at least 55% non-draw win rate over 400 games.
- No failed distillation checkpoints or raw MCTS games enter the package.

---

### Task 1: Deadline-aware belief-PUCT

**Files:**
- Modify: `src/rl/puct.py`
- Modify: `src/rl/belief_puct_agent.py`
- Test: `tests/test_puct.py`
- Test: `tests/test_belief_puct_agent.py`

**Interfaces:**
- `BeliefPUCTSearch.search(..., deadline: float | None = None)` stops between expansions.
- `SearchConfig` adds `time_budget_seconds: float` and `game_budget_seconds: float`.
- `Top2BeliefPUCTAgent` reports `mcts_deadline_fallback` and `mcts_game_budget_fallback` and resets cumulative budget with `reset_trajectory()`.

- [ ] Add failing deterministic-clock tests for expired deadlines and exhausted game budgets.
- [ ] Run the focused tests and verify failures are due to missing controls.
- [ ] Implement monotonic deadline checks, cumulative accounting, and legal frozen-policy fallback.
- [ ] Run the focused tests and verify they pass.

### Task 2: Isolated candidate runtime

**Files:**
- Create: `candidates/primary_budgeted_mcts/main.py`
- Create: `candidates/primary_budgeted_mcts/README.md`
- Test: `tests/test_primary_budgeted_mcts_candidate.py`

**Interfaces:**
- `agent(obs_dict)` returns the primary deck for `None`, resets game budget at game start, and otherwise returns a legal action.
- Environment overrides: `PTCG_MCTS_SIMULATIONS`, `PTCG_MCTS_PARTICLES`, `PTCG_MCTS_MAX_DEPTH`, `PTCG_MCTS_TIME_BUDGET`, `PTCG_MCTS_GAME_BUDGET`.

- [ ] Add failing tests for deck selection, safe defaults, reset, and preservation of formal submission files.
- [ ] Run the candidate tests and confirm the runtime is absent.
- [ ] Implement lazy frozen-policy/search initialization and guarded legal fallback.
- [ ] Run candidate tests and confirm they pass.

### Task 3: Latency and release gate evaluation

**Files:**
- Modify: `scripts/evaluate_top2_mcts.py`
- Create: `scripts/evaluate_primary_budgeted_mcts.py`
- Test: `tests/test_primary_budgeted_mcts_gate.py`

**Interfaces:**
- Evaluator accepts time/game budgets and records action-source counters.
- `submission_gate(report: dict) -> dict` requires 400 games, 55% non-draw rate, p95 <= 0.035, and zero safety failures.

- [ ] Add failing tests for every strength, latency, sample-size, exception, and illegality boundary.
- [ ] Run gate tests and confirm failure because the gate does not exist.
- [ ] Implement budget propagation, source reporting, and pure report gate logic.
- [ ] Run gate and existing resume tests and confirm they pass.

### Task 4: Candidate archive

**Files:**
- Create: `scripts/build_primary_budgeted_mcts_candidate.py`
- Create: `scripts/verify_primary_budgeted_mcts_candidate.py`
- Test: `tests/test_primary_budgeted_mcts_package.py`
- Generate: `server_uploads/pokemon-tcg-primary-budgeted-mcts-v1.tar.gz`
- Generate: `server_uploads/pokemon-tcg-primary-budgeted-mcts-v1.tar.gz.sha256`

**Interfaces:**
- Manifest lists exact file hashes, primary deck/checkpoint identities, runtime defaults, and `formal_submission_replacement_authorized: false`.

- [ ] Add a failing package-content test that rejects reserve assets, raw games, failed candidates, or a changed formal submission.
- [ ] Run the package test and confirm the builder is absent.
- [ ] Implement build and verify scripts and generate the archive from frozen primary inputs.
- [ ] Run package verification, all MCTS tests, and archive SHA256 validation.
