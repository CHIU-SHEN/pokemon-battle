# MCTS Power-Loss Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every arena game durable so the server resumes an interrupted 400-game MCTS evaluation instead of restarting it.

**Architecture:** Add a small checkpoint layer to the evaluator that validates run identity and atomically persists aggregates after each game. Pass resume through the orchestrator, add a sequential server job for the rented single-GPU host, then rebuild and verify the self-contained handoff archive.

**Tech Stack:** Python 3.10+, pytest, PyTorch, Bash, JSON, tar/SHA256.

## Global Constraints

- The instance disk survives restart; remote replication is out of scope.
- Persist after every completed game using temporary-file, flush, fsync, and replace.
- Resume must preserve game index parity and reject incompatible run metadata.
- Arena/MCTS runs on CPU; candidate training runs on CUDA.
- Primary and reserve run sequentially on the rented i9-14900KF/RTX 4060 Ti host.
- Preserve existing report schema and do not authorize promotion or submission replacement.

---

### Task 1: Atomic resumable evaluation state

**Files:**
- Modify: `scripts/evaluate_top2_mcts.py`
- Create: `tests/test_mcts_eval_resume.py`

**Interfaces:**
- Produces: `atomic_write_json(path: Path, payload: dict) -> None`
- Produces: `load_progress(path: Path, identity: dict, resume: bool) -> dict`
- CLI: new `--resume`; progress path is `<output stem>.progress.json`.

- [ ] Write tests that simulate an interruption after two games, assert a parseable two-game checkpoint, resume to four games, and verify game-side parity is `[0, 1, 0, 1]` without replay.
- [ ] Run `python -m pytest tests/test_mcts_eval_resume.py -q` and confirm failure because resume support is absent.
- [ ] Implement run identity, atomic JSON writes, per-game checkpointing, metadata validation, and final atomic report generation.
- [ ] Run `python -m pytest tests/test_mcts_eval_resume.py -q` and confirm all tests pass.

### Task 2: Resume orchestration and sequential server job

**Files:**
- Modify: `scripts/run_top2_mcts_pilot.py`
- Create: `jobs/top2_mcts_pilot_resilient.sh`
- Modify: `TOP2_MCTS_SERVER_HANDOFF.md`
- Modify: `tests/test_top2_mcts_handoff.py`

**Interfaces:**
- Runner passes `--resume` into search and candidate evaluator commands.
- Job accepts `PILOT_ROOT`, `PILOT_GAMES`, and `ARENA_GAMES`, defaulting to `mcts-pilot-200`, `200`, and `400`.

- [ ] Extend handoff tests to require evaluator resume flags and the sequential reliability script.
- [ ] Run the focused handoff tests and confirm they fail for the missing behavior.
- [ ] Pass `--resume` to evaluations; add a strict Bash loop that runs primary then reserve with separate logs and syncs after each branch; document reboot/restart commands.
- [ ] Run the focused handoff tests and confirm they pass.

### Task 3: Build and verify the revised handoff

**Files:**
- Modify: `scripts/build_top2_mcts_handoff.py`
- Generate: `server_uploads/pokemon-tcg-top2-mcts-pilot-v2.tar.gz`
- Generate: `server_uploads/pokemon-tcg-top2-mcts-pilot-v2.tar.gz.sha256`

**Interfaces:**
- Package basename changes to `pokemon-tcg-top2-mcts-pilot-v2` and includes the resilient job.

- [ ] Extend package tests to require the v2 basename and resilient job in the manifest.
- [ ] Run the package test and confirm it fails against v1.
- [ ] Update the package builder, build v2 using the frozen artifacts, and run `scripts/verify_top2_mcts_handoff.py` against an extracted archive.
- [ ] Run all MCTS-focused tests plus SHA256 verification and report the exact archive path and digest.
