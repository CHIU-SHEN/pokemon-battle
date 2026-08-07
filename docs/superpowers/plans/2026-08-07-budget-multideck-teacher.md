# Budget Multideck Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a budget-gated multideck data pipeline that discovers 50 observed decks, selects 12 representative opponent clusters, focuses S128 labels on 4–6 hard clusters, trains one shared network, and produces directly deployable S128 plus S16 fallback candidates within RMB 50.

**Architecture:** Local scripts build and validate an observed-deck catalog, then produce a deterministic opponent schedule. A resumable server runner performs a bounded worker sweep, broad screening, focused S128 collection, shared-network training, and a multideck promotion matrix. S128 and S16 packages share one checkpoint and differ only in runtime search configuration.

**Tech Stack:** Python 3.11, PyTorch, repository `cg` simulator, JSON/JSONL manifests, pytest, Bash single-node runner, existing MCTS and packaging modules.

## Global Constraints

- Total rental budget is at most RMB 50, including compute, bandwidth, and storage charges.
- Active planned compute stops at 15 hours; the absolute paid-instance deadline is 17 hours.
- Training and submission continue to use `crustle_kangaskhan_cage`; opponent decks are training/evaluation diversity, not 50 submission agents.
- S128 is a direct deployment candidate; model compression and S16 distillation are not release prerequisites.
- S16 remains the time-budget fallback and must share the exact selected checkpoint with S128.
- Never serialize hidden opponent cards, prize cards, particles, or full opponent decks into training samples.
- Never overwrite `submission/`, frozen S16 packages, or Cage Rules V1.
- Kaggle upload requires separate explicit authorization.
- All implementation follows test-first red/green cycles and each task ends in a focused commit.

---

## File Structure

- Create `src/cards/observed_decks.py`: pure deck hashing, similarity, clustering, and representative selection.
- Create `scripts/build_observed_deck_catalog.py`: read replay-derived profiles and materialized decks; write Top50 and 12-cluster manifests.
- Create `config/budget_multideck_policy.json`: fixed budgets, cluster counts, worker sweep, mixtures, and promotion gates.
- Create `src/rl/multideck_schedule.py`: deterministic broad/focused game allocation.
- Modify `scripts/collect_top2_mcts.py`: accept an opponent manifest entry and record only public opponent identity metadata.
- Modify `src/rl/mcts_dataset.py`: allow public `opponent_cluster_id` and `opponent_policy_id` metadata while preserving hidden-field rejection.
- Create `scripts/benchmark_mcts_workers.py`: bounded 16/24/32/40 process sweep and stable-worker selection.
- Create `src/rl/budget_guard.py`: elapsed-time, projected-cost, disk, memory, and stop decisions.
- Create `scripts/run_budget_multideck.py`: resumable stage orchestrator with state file and audit gates.
- Modify `scripts/build_mcts_primary_dataset.py`: stratified opponent-cluster accounting and mix manifest.
- Modify `src/rl/mcts_train.py`: cluster-balanced sampler with frozen legacy/new mixture.
- Create `scripts/evaluate_multideck_matrix.py`: compare Cage Rules, current S16, new S16, and new S128 by cluster and seat.
- Modify `scripts/build_v3_s16_candidates.py`: generalize shared-checkpoint packaging to S16 and S128 runtime variants.
- Modify `scripts/verify_v3_s16_candidate.py`: verify arbitrary approved simulation count and shared checkpoint identity.
- Create `jobs/budget_multideck_15h.sh`: single-server Level 1 entrypoint with explicit budget environment.
- Create tests matching every new or modified module.

---

### Task 1: Observed Deck Catalog and Representative Clustering

**Files:**
- Create: `src/cards/observed_decks.py`
- Create: `scripts/build_observed_deck_catalog.py`
- Create: `tests/test_observed_decks.py`
- Create at runtime: `data/observed_decks/top50.json`, `data/observed_decks/clusters12.json`

**Interfaces:**
- Produces: `canonical_deck_hash(card_ids: Sequence[int]) -> str`
- Produces: `multiset_jaccard(left: Sequence[int], right: Sequence[int]) -> float`
- Produces: `select_representatives(rows: Sequence[dict], *, top_n: int, clusters: int, similarity_threshold: float) -> dict`
- Consumes existing complete profiles from `data/external/kaggle_replays/core_combo_candidates.json` and `data/high_score_decks/*/deck.csv`.

- [ ] **Step 1: Write failing hashing, similarity, and deterministic clustering tests**

```python
def test_deck_hash_ignores_order_but_preserves_counts():
    assert canonical_deck_hash([1, 2, 2]) == canonical_deck_hash([2, 1, 2])
    assert canonical_deck_hash([1, 2, 2]) != canonical_deck_hash([1, 1, 2])

def test_representatives_keep_distinct_clusters_and_evidence():
    rows = [
        {"deck_ids": [1] * 60, "observations": 10, "strength": 600.0, "source_ids": ["a"]},
        {"deck_ids": [1] * 59 + [2], "observations": 8, "strength": 590.0, "source_ids": ["b"]},
        {"deck_ids": [3] * 60, "observations": 4, "strength": 620.0, "source_ids": ["c"]},
    ]
    result = select_representatives(rows, top_n=3, clusters=2, similarity_threshold=0.9)
    assert len(result["clusters"]) == 2
    assert sum(item["observations"] for item in result["candidates"]) == 22
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_observed_decks.py -v`

Expected: FAIL because `src.cards.observed_decks` does not exist.

- [ ] **Step 3: Implement pure catalog functions**

```python
def canonical_deck_hash(card_ids):
    payload = "\n".join(str(value) for value in sorted(int(x) for x in card_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def multiset_jaccard(left, right):
    a, b = Counter(left), Counter(right)
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return intersection / union if union else 1.0
```

Implement greedy deterministic clustering ordered by `(-observations, -strength, deck_sha256)`. Reject incomplete decks, illegal decks, unknown card IDs, and any row without source evidence. Label frequency as `observed_sample_count`, never `global_usage`.

- [ ] **Step 4: Implement the catalog CLI**

The CLI must accept `--profiles`, repeated `--deck-root`, `--top-n 50`, `--clusters 12`, `--output-root`, validate every 60-card deck with `check_deck`, and atomically write both manifests.

- [ ] **Step 5: Run focused and existing card tests**

Run: `pytest tests/test_observed_decks.py tests/test_high_score_decks.py tests/test_deck_optimizer.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cards/observed_decks.py scripts/build_observed_deck_catalog.py tests/test_observed_decks.py
git commit -m "feat: build observed multideck catalog"
```

---

### Task 2: Fixed Budget Policy and Deterministic Multideck Schedule

**Files:**
- Create: `config/budget_multideck_policy.json`
- Create: `src/rl/multideck_schedule.py`
- Create: `tests/test_multideck_schedule.py`

**Interfaces:**
- Consumes: `clusters12.json` from Task 1.
- Produces: `build_broad_schedule(clusters: Sequence[dict], games: int, seed: int) -> tuple[OpponentJob, ...]`
- Produces: `build_focused_schedule(screen_report: dict, *, min_clusters: int, max_clusters: int, games: int, seed: int) -> tuple[OpponentJob, ...]`

- [ ] **Step 1: Add the immutable policy file**

```json
{
  "schema_version": "budget_multideck_policy_v1",
  "budget": {"cny_limit": 50.0, "active_hours": 15.0, "absolute_hours": 17.0, "stop_new_work_cny": 45.0},
  "catalog": {"top_n": 50, "clusters": 12, "hard_min": 4, "hard_max": 6},
  "worker_sweep": [16, 24, 32, 40],
  "search": {"screen_simulations": 16, "teacher_simulations": 128, "particles": 3, "max_depth": 10},
  "mixture": {"legacy_min": 0.5, "legacy_max": 0.7},
  "deployment_simulations": [16, 128]
}
```

- [ ] **Step 2: Write failing schedule tests**

Verify all 12 clusters receive both seats, job IDs are stable, focused selection orders by lowest win rate then highest uncertainty, and 4–6 distinct clusters are chosen.

- [ ] **Step 3: Run tests and verify failure**

Run: `pytest tests/test_multideck_schedule.py -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement immutable `OpponentJob` scheduling**

```python
@dataclass(frozen=True)
class OpponentJob:
    game_index: int
    cluster_id: str
    candidate_id: str
    policy_id: str
    learner_side: int
    seed: int
```

Use round-robin allocation, alternate `learner_side`, and derive seeds from the global seed plus game index. Never read hidden opponent deck contents from a game observation.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_multideck_schedule.py -v`

Expected: PASS.

```bash
git add config/budget_multideck_policy.json src/rl/multideck_schedule.py tests/test_multideck_schedule.py
git commit -m "feat: schedule budgeted multideck opponents"
```

---

### Task 3: Public Opponent Metadata and Arbitrary Opponent Collection

**Files:**
- Modify: `src/rl/mcts_dataset.py`
- Modify: `scripts/collect_top2_mcts.py`
- Create: `tests/test_multideck_mcts_collection.py`

**Interfaces:**
- Extend `finalize_mcts_game(..., opponent_cluster_id: str, opponent_policy_id: str)`.
- Collector accepts `--opponent-deck`, `--opponent-policy`, and `--opponent-cluster-id`.
- Stored samples contain public IDs only; `opponent_deck` remains forbidden.

- [ ] **Step 1: Write failing schema and collector-construction tests**

```python
def test_finalized_sample_records_public_opponent_identity_without_deck():
    rows = finalize_mcts_game(
        [VALID_RECORD], game_id="g", branch="primary", deck_id="cage",
        result=0, learner_side=0, checkpoint_sha256="abc",
        opponent_cluster_id="cluster-03", opponent_policy_id="rules:alakazam",
    )
    assert rows[0]["opponent_cluster_id"] == "cluster-03"
    assert "opponent_deck" not in rows[0]
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_multideck_mcts_collection.py -v`

Expected: FAIL because the new arguments are unsupported.

- [ ] **Step 3: Extend sample finalization and collector arguments**

Validate identifiers against `^[a-z0-9][a-z0-9_.:-]{0,127}$`. Load the opponent deck only for simulator setup and never attach it to samples or game documents. Support repository rule agents and frozen adapter policies through one `build_opponent(policy_spec, deck_path)` helper.

- [ ] **Step 4: Run hidden-information and collection tests**

Run: `pytest tests/test_multideck_mcts_collection.py tests/test_mcts_dataset.py tests/test_mcts_collection.py -v`

Expected: PASS, including rejection of serialized `opponent_deck`.

- [ ] **Step 5: Commit**

```bash
git add src/rl/mcts_dataset.py scripts/collect_top2_mcts.py tests/test_multideck_mcts_collection.py
git commit -m "feat: collect public multideck teacher targets"
```

---

### Task 4: Worker Sweep and Hard Budget Guard

**Files:**
- Create: `src/rl/budget_guard.py`
- Create: `scripts/benchmark_mcts_workers.py`
- Create: `tests/test_budget_guard.py`
- Create: `tests/test_mcts_worker_benchmark.py`

**Interfaces:**
- Produces: `BudgetSnapshot(elapsed_hours, projected_cny, disk_free_fraction, memory_used_fraction, swap_bytes)`.
- Produces: `budget_decision(snapshot, policy, *, starting_new_work: bool) -> BudgetDecision`.
- Benchmark output contains `selected_workers`, per-candidate games/hour, errors, fallbacks, RSS, swap, and disk growth.

- [ ] **Step 1: Write failing boundary tests**

Cover exactly 15 hours, 17 hours, RMB45 before new work, RMB50 absolute cost, disk below 20%, nonzero swap, and the stable fastest worker selection.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_budget_guard.py tests/test_mcts_worker_benchmark.py -v`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement pure decisions before process launching**

```python
def budget_decision(snapshot, policy, *, starting_new_work):
    if snapshot.elapsed_hours >= policy.absolute_hours or snapshot.projected_cny >= policy.cny_limit:
        return BudgetDecision(False, True, "absolute_budget")
    if snapshot.disk_free_fraction < 0.20 or snapshot.swap_bytes > 0:
        return BudgetDecision(False, False, "resource_guard")
    if starting_new_work and (snapshot.elapsed_hours >= policy.active_hours or snapshot.projected_cny >= policy.stop_new_work_cny):
        return BudgetDecision(False, False, "stop_new_work")
    return BudgetDecision(True, False, "within_budget")
```

- [ ] **Step 4: Implement bounded worker benchmark**

Each candidate runs the same small S128 workload in a separate output directory. Reject any candidate with exceptions, illegal actions, fallback, swap, memory above 80%, or lower throughput than the previous candidate. Atomically write `worker-benchmark.json`.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_budget_guard.py tests/test_mcts_worker_benchmark.py -v`

Expected: PASS.

```bash
git add src/rl/budget_guard.py scripts/benchmark_mcts_workers.py tests/test_budget_guard.py tests/test_mcts_worker_benchmark.py
git commit -m "feat: enforce rental budget and worker sweep"
```

---

### Task 5: Resumable 15-Hour Stage Orchestrator

**Files:**
- Create: `scripts/run_budget_multideck.py`
- Create: `jobs/budget_multideck_15h.sh`
- Create: `tests/test_budget_multideck_runner.py`

**Interfaces:**
- State schema: `budget_multideck_state_v1` with stages `catalog`, `smoke`, `benchmark`, `screen`, `focus`, `dataset`, `train`, `evaluate`, `package`, `sync`.
- Consumes Tasks 1–4 CLIs and policy JSON.
- Produces one immutable run root with state, logs, manifests, hashes, and retry-safe completion markers.

- [ ] **Step 1: Write failing transition tests**

Test resume skips audited stages, refuses identity mismatch, does not launch focus before screen audit, enters sync at active-hour limit, and returns nonzero at absolute limit.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_budget_multideck_runner.py -v`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement atomic stage state and dry-run commands**

The Python orchestrator must build argument arrays without shell interpolation, persist state through temporary-file replacement, and support `--dry-run`. The Bash wrapper only sets pinned environment variables, checks for duplicate run IDs, and invokes Python; it must not duplicate orchestration logic.

- [ ] **Step 4: Add explicit server preflight**

Require x86_64, Python environment import of torch/numpy, readable `cg` library, at least 100 GiB free disk, at least 96 GiB RAM for more than 24 workers, no existing output writer, and a configured hourly USD price plus USD/CNY rate.

- [ ] **Step 5: Run tests and dry run**

Run: `pytest tests/test_budget_multideck_runner.py -v`

Run: `python scripts/run_budget_multideck.py --config config/budget_multideck_policy.json --run-root experiments/budget-multideck/dry-run --dry-run`

Expected: tests PASS; dry run prints all stages without starting games.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_budget_multideck.py jobs/budget_multideck_15h.sh tests/test_budget_multideck_runner.py
git commit -m "feat: orchestrate 15 hour multideck run"
```

---

### Task 6: Cluster-Balanced Dataset and Shared-Network Training

**Files:**
- Modify: `scripts/build_mcts_primary_dataset.py`
- Modify: `src/rl/mcts_train.py`
- Create: `tests/test_multideck_training.py`

**Interfaces:**
- Dataset manifest records counts by split, cluster, policy, teacher budget, and legacy/new source.
- Produces deterministic weights satisfying legacy fraction `[0.50, 0.70]` and preventing any new opponent cluster from exceeding 25% of new-data weight.

- [ ] **Step 1: Write failing mixture and audit tests**

Use a fixture with one dominant cluster and verify balancing, deterministic weights, frozen split preservation, and rejection of missing cluster IDs in new data.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_multideck_training.py -v`

Expected: FAIL because cluster balancing is not implemented.

- [ ] **Step 3: Implement deterministic source and cluster weights**

Add `compute_multideck_weights(rows, legacy_fraction, seed)` returning one weight per row. Do not alter validation/test samples or infer missing opponent identities. Training writes `best_safe.pt`, `last.pt`, and `summary.json` with the exact selected mixture.

- [ ] **Step 4: Run training-related tests**

Run: `pytest tests/test_multideck_training.py tests/test_mcts_train.py tests/test_mcts_primary_dataset.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mcts_primary_dataset.py src/rl/mcts_train.py tests/test_multideck_training.py
git commit -m "feat: train cluster balanced shared policy"
```

---

### Task 7: Multideck Promotion Matrix and Direct S128 Packaging

**Files:**
- Create: `scripts/evaluate_multideck_matrix.py`
- Modify: `scripts/build_v3_s16_candidates.py`
- Modify: `scripts/verify_v3_s16_candidate.py`
- Create: `tests/test_multideck_matrix.py`
- Modify: `tests/test_v3_s16_candidates.py`

**Interfaces:**
- Evaluation variants: frozen Cage Rules V1, current V3 S16, new S16, new S128.
- Package builder accepts `variants={"s16": SearchConfig(...16...), "s128": SearchConfig(...128...)}` and one checkpoint.
- Both manifests record identical `checkpoint_sha256` and distinct `simulations`.

- [ ] **Step 1: Write failing matrix aggregation and package identity tests**

Verify swapped seats, per-cluster and hard-cluster macro rates, worst-cluster reporting, all safety counters, S128 runtime config, S16 runtime config, and shared checkpoint hash.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_multideck_matrix.py tests/test_v3_s16_candidates.py -v`

Expected: FAIL because the matrix and S128 package variant do not exist.

- [ ] **Step 3: Implement matrix evaluation**

Use one fixed schedule for every variant. Report wins/losses/draws, Wilson interval, both seats, mean/P95, full `mcts_*fallback` count, worst cluster, 12-cluster macro, and hard-cluster macro. Do not promote from aggregate win rate alone.

- [ ] **Step 4: Generalize package builder and verifier**

Rename internal constants without breaking the existing CLI. Produce `pokemon-tcg-multideck-s16.tar.gz` and `pokemon-tcg-multideck-s128.tar.gz`. Raw-exec verification must run both packages without `__file__`, validate the top-level 60-card deck, and confirm no hidden training data is bundled.

- [ ] **Step 5: Run focused and packaging regressions**

Run: `pytest tests/test_multideck_matrix.py tests/test_v3_s16_candidates.py tests/test_primary_budgeted_mcts_kaggle_package.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_multideck_matrix.py scripts/build_v3_s16_candidates.py scripts/verify_v3_s16_candidate.py tests/test_multideck_matrix.py tests/test_v3_s16_candidates.py
git commit -m "feat: evaluate and package direct s128 candidate"
```

---

### Task 8: Full Local Verification and Server Handoff

**Files:**
- Modify: `NEXT_STEPS.md`
- Create at runtime: `server_uploads/budget-multideck-run-v1.tar.gz`
- Create at runtime: `server_uploads/budget-multideck-run-v1.tar.gz.sha256`

**Interfaces:**
- Handoff includes code, config, tests, frozen checkpoint inputs, 12-cluster manifest, job entrypoint, and recovery instructions.
- Excludes raw replay bulk, local caches, prior experiment directories, credentials, and Kaggle tokens.

- [ ] **Step 1: Run the complete relevant test suite**

Run: `pytest tests/test_observed_decks.py tests/test_multideck_schedule.py tests/test_multideck_mcts_collection.py tests/test_budget_guard.py tests/test_mcts_worker_benchmark.py tests/test_budget_multideck_runner.py tests/test_multideck_training.py tests/test_multideck_matrix.py tests/test_v3_s16_candidates.py -v`

Expected: all tests PASS with zero failures.

- [ ] **Step 2: Build the real local catalog**

Run: `python scripts/build_observed_deck_catalog.py --profiles data/external/kaggle_replays/core_combo_candidates.json --deck-root data/high_score_decks --top-n 50 --clusters 12 --output-root data/observed_decks`

Expected: 50 or fewer evidence-backed complete unique candidates, exactly 12 clusters when at least 12 distinct decks exist, and all representatives legal. If fewer than 12 legal distinct candidates exist, stop and acquire additional complete public Episodes locally; do not fabricate decks.

- [ ] **Step 3: Build and verify the handoff archive**

Use a repository build script, not an ad hoc broad archive command. Verify archive member allowlist, SHA-256, clean extraction, dry-run orchestration, and checkpoint hashes.

- [ ] **Step 4: Update handoff documentation**

Record the exact preflight, launch, monitor, resume, sync, and stop commands. State that launch and Kaggle upload remain unperformed and require user authorization.

- [ ] **Step 5: Commit handoff code/documentation only**

```bash
git add NEXT_STEPS.md scripts/build_budget_multideck_handoff.py tests/test_budget_multideck_handoff.py
git commit -m "docs: hand off budget multideck run"
```

- [ ] **Step 6: Execution checkpoint**

Before spending money, present: candidate instance identity, hourly price, exchange rate, projected 17-hour maximum, preflight result, worker-sweep plan, persistent output path, sync destination, and automatic termination method. Wait for explicit launch authorization.

---

## Plan Self-Review

- Spec coverage: catalog, 12 clusters, 4–6 hard clusters, S128 focus, shared training, S128/S16 packaging, safety, budget, worker sweep, recovery, and no automatic Kaggle upload are each assigned to a task.
- Type consistency: `opponent_cluster_id` and `opponent_policy_id` flow unchanged from schedule through collection, dataset, training audit, and evaluation.
- Deployment consistency: S128 and S16 share one checkpoint; only runtime search configuration differs.
- Scope boundary: the plan builds and verifies the workflow locally. Renting, launching the paid run, and uploading Kaggle submissions remain separate authorized actions.
