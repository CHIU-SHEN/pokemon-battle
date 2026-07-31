# Top2 Belief PUCT-MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a small-scale, branch-isolated belief-PUCT MCTS self-play and AlphaZero-style training loop for the frozen Top2 decks.

**Architecture:** A pure PUCT core owns tree statistics and is independent of the game engine. A Search API adapter owns belief-particle state creation and cleanup, while an MCTS agent combines neural priors, neural leaf values, joint-action generation, and visit-count action selection. Dedicated collection, training, evaluation, and handoff entry points preserve the existing primary/reserve state and gated promotion model.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, pytest, official ctypes-backed Pokémon Search API.

## Global Constraints

- First pilot is 10 smoke games, then about 200 games per branch.
- Use 32 simulations, 3 belief particles, maximum depth 8, `c_puct=1.5`.
- Self-play root noise uses `alpha=0.3`, `epsilon=0.25`; Arena disables noise.
- Hidden-state particle contents must never be serialized into features, samples, or reports.
- Search API exceptions and illegal actions must be zero; structured fallback rate must remain below 5%.
- Primary and reserve checkpoints, decks, data roots, and manifests must remain isolated.
- Do not replace `submission/deck.csv` or promote a formal best during the pilot.
- Server jobs must report progress, throughput, fallback rate, and ETA every 10 games.

---

### Task 1: Pure PUCT Statistics and Joint-Action Priors

**Files:**
- Create: `src/rl/puct.py`
- Test: `tests/test_puct.py`

**Interfaces:**
- Produces: `EdgeStats`, `NodeStats`, `puct_score(...)`, `select_puct_action(...)`, `joint_action_priors(...)`, `visit_distribution(...)`, and `mix_dirichlet_noise(...)`.
- Consumes: plain Python mappings, NumPy RNG, and option logits; no Search API dependency.

- [ ] **Step 1: Write failing tests**

Cover zero-visit selection by prior, exploitation after positive backups, normalized joint-action priors, deterministic visit distributions, temperature-zero argmax, and noise normalization.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_puct.py -q`
Expected: FAIL because `src.rl.puct` does not exist.

- [ ] **Step 3: Implement the pure core**

Use:

```python
@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0
```

`puct_score` must implement the formula in the approved design. Joint-action log priors sum selected option log-probabilities, divide by `sqrt(action_length)`, then softmax across candidate actions.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_puct.py -q`
Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rl/puct.py tests/test_puct.py
git commit -m "feat: add pure PUCT search primitives"
```

### Task 2: Search API Lifecycle Adapter

**Files:**
- Create: `src/rl/search_backend.py`
- Test: `tests/test_search_backend.py`
- Modify: `submission/agent/belief.py`

**Interfaces:**
- Consumes: `Observation`, `GameLedger`, `BeliefSampler`, and official `search_begin/search_step/search_release/search_end`.
- Produces: `SearchBackend.begin_particles(...)`, `SearchBackend.step(...)`, `SearchBackend.release(...)`, and `SearchBackend.close()` plus structured counters.

- [ ] **Step 1: Write failing lifecycle tests**

Use injected fake API callables to assert every root and child ID is released exactly once, `close()` is idempotent, invalid particles are counted, and exceptions still call `search_end`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_search_backend.py -q`
Expected: FAIL because the adapter is missing.

- [ ] **Step 3: Implement lifecycle ownership**

Define:

```python
class SearchBackend:
    def begin_particles(
        self, obs: Observation, ledger: GameLedger | None, count: int
    ) -> list[SearchStateRef]: ...
    def step(self, state_id: int, action: tuple[int, ...]) -> SearchStateRef: ...
    def release(self, state_id: int) -> None: ...
    def close(self) -> None: ...
```

The adapter stores only IDs and public observations. Particle card IDs remain local variables and never appear in reports.

- [ ] **Step 4: Run lifecycle and existing search tests**

Run: `python -m pytest tests/test_search_backend.py tests/test_m3_search.py -q`
Expected: pass with zero leaked fake IDs.

- [ ] **Step 5: Commit**

```bash
git add src/rl/search_backend.py submission/agent/belief.py tests/test_search_backend.py
git commit -m "feat: wrap Search API state lifecycle"
```

### Task 3: Neural Belief-PUCT Agent

**Files:**
- Create: `src/rl/belief_puct_agent.py`
- Test: `tests/test_belief_puct_agent.py`
- Modify: `src/rl/top2_rollout.py`

**Interfaces:**
- Consumes: `Top2RolloutAgent` model encoding, `ActionGenerator`, `SearchBackend`, and Task 1 PUCT functions.
- Produces: `BeliefPUCTAgent.choose(...) -> SearchDecision` and an agent-compatible `__call__(obs_dict) -> list[int]`.

- [ ] **Step 1: Write failing behavior tests**

Test forced-action bypass, missing-search-input fallback, 32-simulation budget, depth cap 8, player-switch value sign, same-player no sign change, root-noise toggle, legal chosen action, and fallback samples without policy targets.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_belief_puct_agent.py -q`
Expected: FAIL because the agent is missing.

- [ ] **Step 3: Implement bounded PUCT**

Create immutable `SearchConfig` and `SearchDecision`. Each simulation selects by PUCT, expands through `SearchBackend.step`, evaluates terminal states as `+1/0/-1` and non-terminal leaves with the clipped network value, then backs up with player-aware signs.

- [ ] **Step 4: Verify behavior**

Run: `python -m pytest tests/test_belief_puct_agent.py tests/test_puct.py tests/test_search_backend.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/rl/belief_puct_agent.py src/rl/top2_rollout.py tests/test_belief_puct_agent.py
git commit -m "feat: add neural belief PUCT agent"
```

### Task 4: MCTS Sample Schema and Resumable Collection

**Files:**
- Create: `src/rl/mcts_dataset.py`
- Create: `scripts/collect_top2_mcts.py`
- Test: `tests/test_mcts_dataset.py`
- Test: `tests/test_collect_top2_mcts.py`

**Interfaces:**
- Consumes: `SearchDecision`, stable game splitting, branch state, and opponent schedules.
- Produces: `top2_mcts_sample_v1` game documents and atomic `progress.json`.

- [ ] **Step 1: Write failing schema tests**

Assert visit counts normalize to `policy_target`, terminal `z` is copied to every valid search sample, whole games share a split, fallback decisions are excluded, branch hashes are present, and forbidden particle fields are rejected.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_mcts_dataset.py tests/test_collect_top2_mcts.py -q`
Expected: FAIL for missing modules and script.

- [ ] **Step 3: Implement schema and collector**

Write one JSON per game atomically. `progress.json` must contain completed/target games, samples, nodes, fallbacks, exceptions, illegal actions, elapsed seconds, games/hour, and ETA seconds. Update it every 10 games and support `--resume`.

- [ ] **Step 4: Run a 2-game real-engine collection smoke**

Run:

```bash
python scripts/collect_top2_mcts.py --branch primary --games 2 \
  --simulations 8 --particles 1 --max-depth 3 \
  --output-root artifacts/dev_smoke/mcts_collect
```

Expected: two game files, zero exception/illegal counts, no hidden particle fields.

- [ ] **Step 5: Commit**

```bash
git add src/rl/mcts_dataset.py scripts/collect_top2_mcts.py \
  tests/test_mcts_dataset.py tests/test_collect_top2_mcts.py
git commit -m "feat: collect resumable MCTS self-play targets"
```

### Task 5: AlphaZero-Style Adapter Training

**Files:**
- Create: `src/rl/mcts_train.py`
- Create: `scripts/train_top2_mcts.py`
- Test: `tests/test_mcts_train.py`

**Interfaces:**
- Consumes: `top2_mcts_sample_v1` train/valid samples and current branch best checkpoint.
- Produces: `top2_mcts_checkpoint_v1`, `summary.json`, epoch metrics, and eligibility flags.

- [ ] **Step 1: Write failing loss and loader tests**

Test soft visit-target cross entropy, terminal value MSE without GAE, legal masking, reference KL, train-only loader enforcement, branch isolation, and checkpoint identity.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_mcts_train.py -q`
Expected: FAIL because training code is missing.

- [ ] **Step 3: Implement loss and training CLI**

Use:

```text
policy CE + 1.0 * value MSE + 0.02 * reference KL - 0.005 * entropy
```

Train at `1e-4` for at most 6 epochs with valid policy-CE/value-MSE early stopping. Save only Adapter, policy delta, and value delta plus branch/deck/checkpoint hashes.

- [ ] **Step 4: Run unit tests and one-epoch smoke**

Run: `python -m pytest tests/test_mcts_train.py -q`
Then train the Task 4 smoke output for one epoch and verify finite metrics.

- [ ] **Step 5: Commit**

```bash
git add src/rl/mcts_train.py scripts/train_top2_mcts.py tests/test_mcts_train.py
git commit -m "feat: train adapters from MCTS visit targets"
```

### Task 6: Search and Candidate Arena Evaluation

**Files:**
- Create: `scripts/evaluate_top2_mcts.py`
- Create: `src/rl/mcts_gate.py`
- Test: `tests/test_mcts_gate.py`

**Interfaces:**
- Consumes: pure best, best+MCTS, MCTS candidate, swapped-seat match runner, and latency metrics.
- Produces: search-uplift and distillation-uplift reports with Wilson intervals.

- [ ] **Step 1: Write failing gate tests**

Test the two separate comparisons, 50-game smoke, 400-game minimum, search threshold 55%, candidate threshold 53%, gray-zone extension to 1,000, safety rejection, and no formal state promotion.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_mcts_gate.py -q`
Expected: FAIL for missing gate.

- [ ] **Step 3: Implement evaluation and gate**

Reuse swapped seats and Wilson statistics. Reports must include exceptions, illegal actions, fallback rate, nodes/decision, p50/p95 latency, and checkpoint hashes.

- [ ] **Step 4: Run a 2-game Arena smoke**

Compare pure best and best+MCTS with 8 simulations; verify zero illegal actions and that no state file is promoted.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_top2_mcts.py src/rl/mcts_gate.py tests/test_mcts_gate.py
git commit -m "feat: evaluate MCTS search and distillation uplift"
```

### Task 7: Pilot Orchestration, Handoff, and Full Verification

**Files:**
- Create: `scripts/run_top2_mcts_pilot.py`
- Create: `scripts/build_top2_mcts_handoff.py`
- Create: `scripts/verify_top2_mcts_handoff.py`
- Create: `jobs/top2_mcts_pilot_single_node.sh`
- Create: `TOP2_MCTS_SERVER_HANDOFF.md`
- Modify: `项目进度.md`
- Test: `tests/test_top2_mcts_handoff.py`

**Interfaces:**
- Consumes: Tasks 3-6.
- Produces: one resumable 10→200 game pilot command per branch and a checksum-verified server archive.

- [ ] **Step 1: Write failing orchestration and package tests**

Assert branch isolation, smoke-before-pilot enforcement, progress every 10 games, no automatic 3,000-game expansion, required frozen artifacts, and manifest SHA verification.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_top2_mcts_handoff.py -q`
Expected: FAIL for missing scripts and handoff files.

- [ ] **Step 3: Implement runner and single-node job**

The runner stages are collect, train, search-smoke, candidate-smoke, and 400-game Arena. Every stage is atomic and resumable. Any exception, illegal action, or fallback rate ≥5% stops the pilot.

- [ ] **Step 4: Execute end-to-end local smoke**

Run primary with 2 collection games, 8 simulations, 1 particle, depth 3, one train epoch, and 2 Arena games. Confirm zero exceptions/illegal actions and expected smoke-only non-promotion.

- [ ] **Step 5: Run full verification**

Run:

```bash
python -m pytest tests -q
python -m compileall src scripts
python scripts/build_top2_mcts_handoff.py
python scripts/verify_top2_mcts_handoff.py
```

Expected: all tests pass, compilation succeeds, manifest has no missing/mismatched files.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_top2_mcts_pilot.py scripts/build_top2_mcts_handoff.py \
  scripts/verify_top2_mcts_handoff.py jobs/top2_mcts_pilot_single_node.sh \
  TOP2_MCTS_SERVER_HANDOFF.md 项目进度.md tests/test_top2_mcts_handoff.py
git commit -m "build: prepare Top2 MCTS pilot handoff"
```

