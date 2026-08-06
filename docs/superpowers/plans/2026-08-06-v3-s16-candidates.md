# V3 S16 Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the recovered V3 results and build separately named, checksum-verified authority-S16 and Kaggle-S16-60ms candidates that explicitly load the epoch 44 student checkpoint.

**Architecture:** A shared V3 candidate entry point reads a packaged runtime JSON and explicitly passes the packaged Arena checkpoint to `Top2RolloutAgent`. One builder emits an authority package with the proven 250 ms/120 s limits and a flat Kaggle package with 60 ms/3 s limits; a verifier checks identities, checkpoint provenance, budgets, hashes, deck layout, and non-promotion status.

**Tech Stack:** Python 3, PyTorch checkpoints, tarfile/SHA-256, pytest, existing belief-PUCT runtime.

## Global Constraints

- Both candidates use V3 epoch 44 `best_safe_arena.pt` and 16 simulations, 3 particles, depth 10.
- Authority limits are 0.25 seconds per decision and 120 seconds per game.
- Kaggle limits are 0.06 seconds per decision and 5 seconds per game.
- `formal_submission_replacement_authorized` remains `false`.
- Kaggle package remains not-ready until its new 100/400-game gates pass.
- Do not modify or delete the two existing untracked V2 report JSON files.

---

### Task 1: Archive and promote recovered evidence

**Files:**
- Move: `mcts-teacher-v3-results.tar.gz*` to `artifacts/mcts_teacher_v3/primary-5k/archives/`
- Move: `mcts-v3-hybrid-eval-results.tar.gz*` to `artifacts/mcts_teacher_v3/primary-5k/archives/`
- Create: `reports/mcts_teacher_v3_s0_100.json`
- Create: `reports/mcts_teacher_v3_s8_100.json`
- Create: `reports/mcts_teacher_v3_s16_100.json`
- Create: `reports/mcts_teacher_v3_s8_400.json`
- Create: `reports/mcts_teacher_v3_s16_400.json`

**Interfaces:**
- Consumes: downloaded server archives and checksum files.
- Produces: organized immutable archives and stable formal report paths used by documentation and package manifests.

- [ ] **Step 1: Verify both downloaded SHA-256 files**

Use `Get-FileHash` and require exact equality with each `.sha256` first field.

- [ ] **Step 2: Inspect archive members before extraction**

Require the hybrid archive to contain the five named evaluation JSON files and `train/best_safe_arena.pt`.

- [ ] **Step 3: Extract the hybrid archive into a temporary cache directory**

Use `.cache/v3-hybrid-import/`; never extract directly over the authoritative asset tree.

- [ ] **Step 4: Copy formal JSON reports and compare checkpoint hashes**

Copy the five final JSON files to the stable `reports/` names and require the downloaded Arena checkpoint hash to match the locally derived epoch 44 Arena checkpoint.

- [ ] **Step 5: Move the four downloaded archive/checksum files into `archives/`**

Verify the repository root no longer contains either V3 archive basename.

### Task 2: Build the shared V3 S16 runtime with TDD

**Files:**
- Create: `candidates/v3_s16/main.py`
- Create: `scripts/build_v3_s16_candidates.py`
- Create: `scripts/verify_v3_s16_candidate.py`
- Create: `tests/test_v3_s16_candidates.py`

**Interfaces:**
- Consumes: `runtime_config.json`, `model/best_safe_arena.pt`, existing shared/adapter weights, deck assets, and belief-PUCT modules.
- Produces: `agent(obs_dict)` plus `build_all(output_dir, code_root, frozen_root) -> dict[str, Path | dict]` and `verify(root: Path, expected_variant: str) -> dict`.

- [ ] **Step 1: Write a failing package test**

The test builds both variants in `tmp_path` and asserts distinct archive names, exact runtime budgets, checkpoint source epoch 44, flat top-level Kaggle files, complete hashes, and both authorization/readiness flags are false.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python -m pytest tests/test_v3_s16_candidates.py -q`

Expected: failure because the builder module does not exist.

- [ ] **Step 3: Implement the minimal shared runtime**

`main.py` loads `runtime_config.json`, constructs `Top2RolloutAgent(..., ppo_checkpoint=root / "model/best_safe_arena.pt")`, wraps it in `Top2BeliefPUCTAgent`, validates legal output, and falls back to the V3 student policy on exceptions. The evaluation helper counts every `mcts_*fallback` action source, including deadline, game-budget, exception, and illegal-action fallbacks.

- [ ] **Step 4: Implement the dual builder and verifier**

Emit `pokemon-tcg-v3-s16-authority.tar.gz` with a wrapper directory and `pokemon-tcg-v3-s16-kaggle-60ms.tar.gz` as a flat Kaggle archive. Include manifests, exact runtime JSON, top-level Kaggle `main.py`/`deck.csv`/`cg`, the V3 Arena checkpoint, and all runtime dependencies.

- [ ] **Step 5: Run the focused test and confirm GREEN**

Run: `python -m pytest tests/test_v3_s16_candidates.py -q`

Expected: all tests pass.

### Task 3: Build, inspect, and smoke both deliverables

**Files:**
- Create: `final_submissions/pokemon-tcg-v3-s16-authority.tar.gz`
- Create: `final_submissions/pokemon-tcg-v3-s16-authority.tar.gz.sha256`
- Create: `final_submissions/pokemon-tcg-v3-s16-kaggle-60ms.tar.gz`
- Create: `final_submissions/pokemon-tcg-v3-s16-kaggle-60ms.tar.gz.sha256`
- Create: `reports/mcts_teacher_v3_s16_candidate_report.md`

**Interfaces:**
- Consumes: Task 2 builder, verifier, and Task 1 evidence.
- Produces: reproducible candidate archives and a reviewed decision record.

- [ ] **Step 1: Build both archives**

Run: `python scripts/build_v3_s16_candidates.py --output-dir final_submissions`

- [ ] **Step 2: Verify checksums and extracted manifests**

Extract each archive under `.cache/v3-s16-verify/` and run `scripts/verify_v3_s16_candidate.py` with the corresponding expected variant.

- [ ] **Step 3: Run raw-exec/runtime smoke**

Execute the flat Kaggle `main.py` without relying on `__file__`, request the 60-card deck, construct the S16 runtime, and confirm the V3 checkpoint loads.

- [ ] **Step 4: Run focused and full tests**

Run the V3 candidate tests, existing primary package tests, and then `python -m pytest -q` with `PTCG_FROZEN_SOURCE_ROOT` set to the repository root.

- [ ] **Step 5: Write the candidate report**

Record S8/S16 100/400 results, selected S16 configuration, archive checksums, verification results, and the explicit requirement that 60ms must pass a new 100-game gate before upload.

- [ ] **Step 6: Commit only source, tests, plans, and reviewed reports**

Do not add ignored binary archives or unrelated untracked V2 reports.
