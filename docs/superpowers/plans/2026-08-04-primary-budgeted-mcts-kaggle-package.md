# Primary Budgeted MCTS Kaggle Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a top-level Kaggle `.tar.gz` submission for the validated primary budgeted-MCTS candidate and document the release evidence.

**Architecture:** Reuse the already verified server-candidate file selection, but stage files directly at the archive root and copy the primary deck to top-level `deck.csv`. A dedicated verifier enforces the Kaggle layout, hashes, primary-only identity, and runtime entry point without changing `submission/`.

**Tech Stack:** Python 3.11, `tarfile`, `hashlib`, pytest, Markdown.

## Global Constraints

- Output is `final_submissions/primary_budgeted_mcts_v1.tar.gz`.
- `main.py` and `deck.csv` are at archive root with no wrapper directory.
- Runtime defaults are 8 simulations, 1 particle, depth 4, 30 ms per decision, and 2 seconds per game.
- Do not include reserve assets, raw games, training data, or evaluation logs.
- Do not replace `submission/` or upload to Kaggle.

---

### Task 1: Lock the Kaggle archive contract

**Files:**
- Create: `tests/test_primary_budgeted_mcts_kaggle_package.py`
- Create: `scripts/build_primary_budgeted_mcts_kaggle.py`

**Interfaces:**
- Produces: `build(output_dir: Path, code_root: Path, frozen_root: Path) -> tuple[Path, Path, dict]`

- [ ] Write a failing test requiring top-level `main.py`, `deck.csv`, `cg/`, model assets and manifest, with no wrapper or reserve files.
- [ ] Run the test and confirm failure because the builder module does not exist.
- [ ] Implement the minimal deterministic staging, manifest, tar creation, and SHA-256 sidecar.
- [ ] Run the focused test and confirm it passes.

### Task 2: Add independent verification

**Files:**
- Create: `scripts/verify_primary_budgeted_mcts_kaggle.py`
- Modify: `tests/test_primary_budgeted_mcts_kaggle_package.py`

**Interfaces:**
- Produces: `verify(root: Path) -> dict`

- [ ] Add a failing test for hash verification, top-level deck parity, 60-card `agent(None)`, and reserve-asset rejection.
- [ ] Run the focused test and confirm the verifier import or checks fail.
- [ ] Implement the verifier and include it in the package.
- [ ] Run focused package tests and confirm they pass.

### Task 3: Build the release and update documentation

**Files:**
- Modify: `README.md`
- Modify: `项目进度.md`
- Create: `reports/primary_budgeted_mcts_v1_submission_report.md`

**Interfaces:**
- Consumes: verified archive path, checksum, 400-game evaluation and submission gate JSON.

- [ ] Correct the README submission format to the current Kaggle `.tar.gz` top-level contract.
- [ ] Record the 261-139 result, 65.25% win rate, 30.09 ms p95, zero safety failures, and package distinction.
- [ ] Build the archive from frozen primary weights.
- [ ] Extract to a fresh directory, run the verifier and load `agent(None)`.
- [ ] Run the complete `tests` suite and `git diff --check`.
