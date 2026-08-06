# Unified Cache Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove regenerable root-level caches, route future pytest state into `.cache/pytest/`, and conservatively ignore other verified generated outputs.

**Architecture:** Pytest owns `.cache/pytest/cache` and `.cache/pytest/tmp` through repository configuration. Git ignores the complete `.cache/` tree plus legacy cache names, runtime progress/log files, and downloaded result bundles while leaving final JSON reports visible for review.

**Tech Stack:** pytest configuration, Git ignore patterns, PowerShell filesystem checks.

## Global Constraints

- Delete only `.pytest-*`, `.pytest_cache`, `.test-tmp-*`, and `.kaggle-notebook-inspect` at the repository root.
- Do not delete or move files under `artifacts/` or `reports/`.
- Keep final `reports/*.json` and Markdown summaries visible to Git unless they are explicitly runtime progress files.
- Existing tracked files remain tracked.

---

### Task 1: Centralize pytest state

**Files:**
- Modify: `pytest.ini`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: pytest's `cache_dir` setting and `--basetemp` option.
- Produces: `.cache/pytest/cache/` and `.cache/pytest/tmp/` as the only repository-local pytest state roots.

- [ ] **Step 1: Capture the baseline configuration and root cache inventory**

Run: `Get-Content pytest.ini; Get-ChildItem -Force -Directory | Where-Object Name -Match '^\.(pytest|test-tmp|kaggle-notebook-inspect)'`

Expected: `pytest.ini` disables the cache provider and the listed legacy directories are visible.

- [ ] **Step 2: Configure pytest's unified cache paths**

Track an empty `.cache/pytest/.gitkeep` directory anchor, then set `pytest.ini` to:

```ini
[pytest]
testpaths = tests
cache_dir = .cache/pytest/cache
addopts = --basetemp=.cache/pytest/tmp
```

- [ ] **Step 3: Add cache and generated-output ignore rules**

Add these scoped rules to `.gitignore`:

```gitignore
# Unified repository-local tool caches
.cache/*
!.cache/pytest/
.cache/pytest/*
!.cache/pytest/.gitkeep
.test-tmp-*/
.kaggle-notebook-inspect/

# Downloaded/reproducible MCTS result bundles
artifacts/*-results.tar.gz
artifacts/*-results.tar.gz.sha256
artifacts/mcts_teacher_v2/

# Runtime evaluation state; final JSON reports remain reviewable
reports/*.log
reports/*.progress.json
```

- [ ] **Step 4: Validate the configuration diff**

Run: `git diff --check -- pytest.ini .gitignore`

Expected: exit code 0.

### Task 2: Remove legacy caches and verify prevention

**Files:**
- Delete: root `.pytest-*`, `.pytest_cache`, `.test-tmp-*`, `.kaggle-notebook-inspect`
- Test: `tests/test_mcts_teacher_v3_handoff.py`
- Test: `tests/test_mcts_teacher_v3_job.py`

**Interfaces:**
- Consumes: unified pytest configuration from Task 1.
- Produces: clean repository root and regenerated state only below `.cache/pytest/`.

- [ ] **Step 1: Resolve and verify every deletion target**

Enumerate matching root directories and confirm every resolved path has the repository root as its direct parent.

- [ ] **Step 2: Delete only the verified cache targets**

Use PowerShell `Remove-Item -LiteralPath <verified-path> -Recurse -Force` for each resolved directory.

- [ ] **Step 3: Run the focused test suite without command-line cache overrides**

Run: `python -m pytest tests/test_mcts_teacher_v3_handoff.py tests/test_mcts_teacher_v3_job.py -q`

Expected: `2 passed`.

- [ ] **Step 4: Verify filesystem and Git behavior**

Confirm no legacy root cache directory exists, `.cache/pytest/` exists, generated bundles/progress/logs are ignored, and final report JSON files remain visible in `git status`.

- [ ] **Step 5: Commit the cleanup configuration**

```powershell
git add -- pytest.ini .gitignore
git commit -m "chore: centralize local cache files"
```
