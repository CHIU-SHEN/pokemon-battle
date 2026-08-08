# Kaggle Full-Leaderboard Deck Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable breadth-first Kaggle pipeline that scans public teams, samples three episodes per latest active submission, extracts and validates both 60-card decks, deduplicates them, and creates non-fixed deck-family clusters.

**Architecture:** Add a focused `src/deck_discovery` package whose pure parsing/catalog functions are independent of Kaggle and whose client is injected behind a protocol for deterministic tests. A thin CLI orchestrates checkpointed stages and temporary replay cleanup. Exact deck identity uses the repository's existing sorted-card SHA-256 convention; clustering uses sparse count vectors plus HDBSCAN with noise preservation.

**Tech Stack:** Python 3.10+, Kaggle Python API already present in the training environment, standard-library JSON/CSV/hash/path utilities, scikit-learn sparse preprocessing, `hdbscan`, pytest.

## Global Constraints

- Do not modify or delete existing `data/external/kaggle_replays/raw`, training data, MCTS code, Adapter code, or user artifacts.
- Default output is `data/deck_discovery/`; temporary replay files live only under its explicit `tmp_replays/` child.
- Default discovery is the latest active submission and three completed episodes per team.
- Default successful-request delay is 2 seconds; retry at most 6 times; HTTP 429 backoff is capped at 300 seconds.
- A replay is deleted only after both valid deck observations and checkpoint state are durably written and re-read successfully.
- Scores are metadata only and never enter clustering distance.
- Clustering must choose its own cluster count, allow noise, expose confidence, and use a real observed medoid as representative.
- Full-leaderboard or server collection must not start without notifying the user and receiving authorization; only a 20-team local smoke is permitted during implementation verification.

---

## File Map

- Create `src/deck_discovery/__init__.py`: public package exports.
- Create `src/deck_discovery/models.py`: typed records, stable IDs, serialization and schema constants.
- Create `src/deck_discovery/replay.py`: complete two-deck extraction, result metadata and validation.
- Create `src/deck_discovery/catalog.py`: idempotent checkpoint store, exact deck aggregation and atomic exports.
- Create `src/deck_discovery/kaggle_client.py`: Kaggle API protocol, authenticated adapter, pagination and retry policy.
- Create `src/deck_discovery/pipeline.py`: breadth-first stage orchestration and adaptive follow-up selection.
- Create `src/deck_discovery/clustering.py`: sparse features, HDBSCAN fit, confidence and medoid selection.
- Create `scripts/discover_kaggle_decks.py`: CLI only; no domain logic.
- Create `tests/test_deck_discovery_replay.py`: replay extraction and validation tests.
- Create `tests/test_deck_discovery_catalog.py`: idempotency, atomicity and recovery tests.
- Create `tests/test_deck_discovery_client.py`: pagination, retry and selection tests with fakes.
- Create `tests/test_deck_discovery_pipeline.py`: end-to-end fake-client orchestration tests.
- Create `tests/test_deck_discovery_clustering.py`: exact, variant, distinct-family and noise tests.
- Modify `requirements-train.txt`: add the pinned clustering dependency only after the clustering test requires it.
- Modify `data/external/README.md`: document dry-run, 20-team smoke, resume and the authorization gate for full scans.

### Task 1: Stable records and replay extraction

**Files:**
- Create: `src/deck_discovery/__init__.py`
- Create: `src/deck_discovery/models.py`
- Create: `src/deck_discovery/replay.py`
- Test: `tests/test_deck_discovery_replay.py`

**Interfaces:**
- Produces: `canonical_deck_hash(cards: Sequence[int]) -> str`
- Produces: `extract_replay(path: Path, cards_db: dict, tags: dict) -> list[DeckObservation]`
- Produces: frozen dataclass `DeckObservation(observation_id, episode_id, player_index, team_name, cards, deck_hash, result)` with `to_dict()`.

- [ ] **Step 1: Write failing canonicalization and two-player extraction tests**

```python
def test_extract_replay_returns_two_valid_60_card_observations(tmp_path, cards_and_tags):
    replay = replay_with_action_decks([list(range(1, 61)), list(range(61, 121))])
    path = tmp_path / "episode-42-replay.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    rows = extract_replay(path, *cards_and_tags)
    assert [row.player_index for row in rows] == [0, 1]
    assert all(len(row.cards) == 60 for row in rows)
    assert rows[0].deck_hash == canonical_deck_hash(reversed(rows[0].cards))

def test_extract_replay_rejects_incomplete_deck(tmp_path, cards_and_tags):
    path = write_replay(tmp_path, replay_with_action_decks([[1] * 59, [2] * 60]))
    with pytest.raises(ValueError, match="exactly 60"):
        extract_replay(path, *cards_and_tags)
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `pytest -q tests/test_deck_discovery_replay.py`
Expected: FAIL during import because `src.deck_discovery.replay` does not exist.

- [ ] **Step 3: Implement minimal typed extraction using both supported replay encodings**

```python
def canonical_deck_hash(cards: Sequence[int]) -> str:
    canonical = "\n".join(str(card_id) for card_id in sorted(map(int, cards)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def raw_decks(replay: dict[str, Any]) -> list[list[int]]:
    action = replay["steps"][0][0]["visualize"][0].get("action")
    if isinstance(action, list) and len(action) == 2:
        return [[int(card_id) for card_id in deck] for deck in action]
    for step in replay.get("steps", []):
        for record in step:
            for frame in record.get("visualize", []) if isinstance(record, dict) else []:
                players = frame.get("current", {}).get("players", [])
                if len(players) == 2:
                    decks = [[int(card["id"]) for card in player.get("deck", [])] for player in players]
                    if all(len(deck) == 60 for deck in decks):
                        return decks
    raise ValueError("no two complete 60-card decks")
```

Call existing `check_deck(cards, cards_db, tags)` for each deck and raise a structured `ValueError` containing episode and player when invalid. Derive result from terminal rewards and team names from `info.TeamNames`.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_deck_discovery_replay.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/deck_discovery tests/test_deck_discovery_replay.py
git commit -m "feat: extract validated replay decks"
```

### Task 2: Idempotent catalog and atomic exports

**Files:**
- Create: `src/deck_discovery/catalog.py`
- Test: `tests/test_deck_discovery_catalog.py`

**Interfaces:**
- Consumes: `DeckObservation.to_dict()` and `canonical_deck_hash` from Task 1.
- Produces: `DiscoveryCatalog(root: Path)` with `upsert_team`, `upsert_submission`, `upsert_episode`, `commit_observations`, `export`, and `summary`.
- Produces: JSONL files defined in the design; every upsert is keyed and idempotent.

- [ ] **Step 1: Write failing idempotency and atomic-recovery tests**

```python
def test_repeated_observation_is_idempotent(tmp_path, observation):
    catalog = DiscoveryCatalog(tmp_path)
    catalog.commit_observations([observation])
    catalog.commit_observations([observation])
    catalog.export()
    assert len(read_jsonl(tmp_path / "decks.jsonl")) == 1
    assert len(read_jsonl(tmp_path / "unique_decks.jsonl")) == 1

def test_failed_export_preserves_previous_file(tmp_path, monkeypatch, observation):
    catalog = DiscoveryCatalog(tmp_path)
    catalog.commit_observations([observation]); catalog.export()
    before = (tmp_path / "decks.jsonl").read_bytes()
    monkeypatch.setattr(Path, "replace", raising_replace)
    with pytest.raises(OSError): catalog.export()
    assert (tmp_path / "decks.jsonl").read_bytes() == before
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_deck_discovery_catalog.py`
Expected: FAIL because `DiscoveryCatalog` is undefined.

- [ ] **Step 3: Implement keyed in-memory loading plus atomic JSONL replacement**

```python
def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
```

On initialization, reload existing JSONL by unique key. `commit_observations` must aggregate all sources under the exact deck hash without overwriting previous provenance. Export then re-read affected files before an episode may be marked `processed`.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_deck_discovery_catalog.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/deck_discovery/catalog.py tests/test_deck_discovery_catalog.py
git commit -m "feat: add resumable deck discovery catalog"
```

### Task 3: Kaggle discovery client and bounded retry

**Files:**
- Create: `src/deck_discovery/kaggle_client.py`
- Test: `tests/test_deck_discovery_client.py`

**Interfaces:**
- Produces protocol `DiscoveryClient.list_teams()`, `list_submissions(team_id)`, `list_episodes(submission_id)`, `download_replay(episode_id, destination)`.
- Produces `KaggleDiscoveryClient(api, delay_seconds=2.0, attempts=6, sleep=time.sleep)`.
- Produces pure selectors `latest_active_submission(rows)` and `recent_completed_episodes(rows, limit)`.

- [ ] **Step 1: Write failing pagination, filtering and 429 tests**

```python
def test_list_teams_follows_page_tokens(fake_api):
    fake_api.pages = [([{"teamId": 1}], "next"), ([{"teamId": 2}], None)]
    assert [x["team_id"] for x in KaggleDiscoveryClient(fake_api, sleep=lambda _: None).list_teams()] == [1, 2]

def test_retry_caps_429_delay(fake_api):
    fake_api.failures = [TooManyRequests(), TooManyRequests(), None]
    sleeps = []
    client = KaggleDiscoveryClient(fake_api, sleep=sleeps.append)
    client.list_submissions(7)
    assert sleeps == [60, 120]
    assert max(sleeps) <= 300
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_deck_discovery_client.py`
Expected: FAIL because the client module does not exist.

- [ ] **Step 3: Implement adapter and selectors**

Normalize SDK objects immediately into plain dictionaries. Continue pagination until the returned token is empty. Recognize 429 from status codes or Kaggle-wrapped exception text; use delays `60, 120, 180, 240, 300`. For other retryable errors use `1, 2, 4, 8, 16`; the sixth failure is recorded and raised. Do not retry authentication/permission errors.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_deck_discovery_client.py`
Expected: PASS without network access.

- [ ] **Step 5: Commit**

```powershell
git add src/deck_discovery/kaggle_client.py tests/test_deck_discovery_client.py
git commit -m "feat: enumerate public Kaggle deck sources"
```

### Task 4: Breadth-first resumable pipeline

**Files:**
- Create: `src/deck_discovery/pipeline.py`
- Test: `tests/test_deck_discovery_pipeline.py`

**Interfaces:**
- Consumes: `DiscoveryClient`, `DiscoveryCatalog`, `extract_replay`.
- Produces: `DiscoveryConfig(max_teams, episodes_per_submission=3, keep_replays=False, dry_run=False)`.
- Produces: `run_discovery(client, catalog, config, cards_db, tags) -> dict[str, Any]`.

- [ ] **Step 1: Write failing end-to-end fake-client tests**

```python
def test_pipeline_deduplicates_shared_episode_and_removes_success(tmp_path, fake_client, rules):
    fake_client.given_two_teams_sharing_episode(99)
    report = run_discovery(fake_client, DiscoveryCatalog(tmp_path), DiscoveryConfig(max_teams=2), *rules)
    assert fake_client.download_calls == [99]
    assert report["valid_deck_observations"] == 2
    assert not (tmp_path / "tmp_replays" / "episode-99-replay.json").exists()

def test_pipeline_keeps_failed_replay_and_resumes(tmp_path, fake_client, rules):
    fake_client.given_corrupt_episode(100)
    run_discovery(fake_client, DiscoveryCatalog(tmp_path), DiscoveryConfig(max_teams=1), *rules)
    assert (tmp_path / "failures" / "episode-100-replay.json").exists()
    first_calls = list(fake_client.download_calls)
    run_discovery(fake_client, DiscoveryCatalog(tmp_path), DiscoveryConfig(max_teams=1), *rules)
    assert fake_client.download_calls == first_calls
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_deck_discovery_pipeline.py`
Expected: FAIL because pipeline interfaces are absent.

- [ ] **Step 3: Implement stage orchestration and state transitions**

Use explicit episode states `scheduled`, `downloaded`, `processed`, `failed_retryable`, `failed_terminal`. Persist after each team, submission and episode. A successful path is download → extract two observations → validate → commit/export/re-read → mark processed → delete exact temporary file. A failure path moves the exact file into `failures/` and records exception type/message without deleting unrelated files.

- [ ] **Step 4: Add adaptive follow-up selection tests and implementation**

```python
def test_unstable_submission_is_extended_to_ten_episodes():
    hashes = ["a", "b", "a"]
    assert follow_up_limit(hashes, score=900) == 10

def test_stable_submission_needs_no_follow_up():
    assert follow_up_limit(["a", "a", "a"], score=1200) == 3
```

High-score historical-submission discovery is emitted as a second-stage recommendation, not automatically downloaded during first-pass smoke.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest -q tests/test_deck_discovery_pipeline.py`
Expected: PASS.

```powershell
git add src/deck_discovery/pipeline.py tests/test_deck_discovery_pipeline.py
git commit -m "feat: orchestrate resumable deck discovery"
```

### Task 5: Dynamic clustering with noise and real medoids

**Files:**
- Create: `src/deck_discovery/clustering.py`
- Test: `tests/test_deck_discovery_clustering.py`
- Modify: `requirements-train.txt`

**Interfaces:**
- Consumes: exact deck records from `unique_decks.jsonl`.
- Produces: `cluster_decks(decks, min_cluster_size=5, min_samples=None) -> list[ClusterAssignment]`.
- Produces: `ClusterAssignment(deck_hash, cluster_id, probability, is_noise, representative_hash)`.

- [ ] **Step 1: Write failing synthetic-family tests**

```python
def test_dynamic_clusters_keep_outlier_as_noise_and_use_real_medoid():
    decks = two_dense_families_plus_outlier()
    rows = cluster_decks(decks, min_cluster_size=3)
    stable = {row.cluster_id for row in rows if not row.is_noise}
    assert len(stable) == 2
    assert sum(row.is_noise for row in rows) == 1
    hashes = {deck["deck_hash"] for deck in decks}
    assert all(row.representative_hash in hashes for row in rows if not row.is_noise)
```

- [ ] **Step 2: Run and verify dependency/module failure**

Run: `pytest -q tests/test_deck_discovery_clustering.py`
Expected: FAIL because clustering code or `hdbscan` is missing.

- [ ] **Step 3: Add pinned dependency and implement sparse features**

Add a compatible exact `hdbscan` version to `requirements-train.txt`. Build a sparse card-count matrix, L2-normalize it for cosine geometry, fit HDBSCAN without a fixed cluster count, preserve label `-1` as noise, and expose `probabilities_`. Select each medoid from members by minimum total weighted-Jaccard distance.

- [ ] **Step 4: Run tests and check score exclusion**

Run: `pytest -q tests/test_deck_discovery_clustering.py`
Expected: PASS, including a test proving score changes do not change the feature matrix.

- [ ] **Step 5: Commit**

```powershell
git add requirements-train.txt src/deck_discovery/clustering.py tests/test_deck_discovery_clustering.py
git commit -m "feat: cluster discovered deck families"
```

### Task 6: CLI, summary and safe local smoke

**Files:**
- Create: `scripts/discover_kaggle_decks.py`
- Modify: `data/external/README.md`
- Test: `tests/test_deck_discovery_cli.py`

**Interfaces:**
- Consumes all preceding modules.
- Produces CLI flags from the design and commands `discover`, `cluster`, `report`.
- Produces `summary.json` with coverage, failure, score-band, unique-deck, cluster, noise and marginal-yield metrics.

- [ ] **Step 1: Write failing CLI contract tests**

```python
def test_cli_defaults_are_safe():
    args = parser().parse_args(["discover", "--dry-run"])
    assert args.episodes_per_submission == 3
    assert args.delay_seconds == 2.0
    assert args.dry_run is True

def test_full_scan_requires_explicit_acknowledgement():
    with pytest.raises(SystemExit):
        main(["discover"])
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_deck_discovery_cli.py`
Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement thin CLI and full-scan safety gate**

Require either `--max-teams N` or an explicit `--ack-full-scan` flag. `--ack-full-scan` only removes the CLI guard; project procedure still requires notifying the user before starting. Dry-run may enumerate/plan but must not download replay files.

- [ ] **Step 4: Document exact local commands**

```powershell
python scripts/discover_kaggle_decks.py discover --dry-run --max-teams 20
python scripts/discover_kaggle_decks.py discover --max-teams 20 --episodes-per-submission 3
python scripts/discover_kaggle_decks.py cluster
python scripts/discover_kaggle_decks.py report
```

State explicitly that no server or full 6,519-team run begins without user approval.

- [ ] **Step 5: Run offline CLI tests and commit**

Run: `pytest -q tests/test_deck_discovery_cli.py`
Expected: PASS.

```powershell
git add scripts/discover_kaggle_decks.py data/external/README.md tests/test_deck_discovery_cli.py
git commit -m "feat: add safe deck discovery CLI"
```

### Task 7: Regression verification and 20-team smoke gate

**Files:**
- Modify only if a verified defect is found in files created by Tasks 1–6.
- Generate ignored/runtime output only under `data/deck_discovery/`; do not commit collected data.

**Interfaces:**
- Verifies the complete feature; produces no new production API.

- [ ] **Step 1: Run all new tests**

Run: `pytest -q tests/test_deck_discovery_replay.py tests/test_deck_discovery_catalog.py tests/test_deck_discovery_client.py tests/test_deck_discovery_pipeline.py tests/test_deck_discovery_clustering.py tests/test_deck_discovery_cli.py`
Expected: PASS.

- [ ] **Step 2: Run relevant existing regression tests**

Run: `pytest -q tests/test_kaggle_replay_conversion.py tests/test_high_score_decks.py tests/test_adapter_sampling_views.py tests/test_deck_optimizer.py`
Expected: PASS.

- [ ] **Step 3: Run full repository tests with a safe temporary base**

```powershell
$testTemp = Join-Path $env:TEMP ("ptcg-tests-" + [guid]::NewGuid())
pytest -q --basetemp $testTemp
```

Expected: PASS with no new failures.

- [ ] **Step 4: Run a network dry-run for 20 teams**

Run: `python scripts/discover_kaggle_decks.py discover --dry-run --max-teams 20`
Expected: 20 teams enumerated, latest public submissions selected where available, no replay file created.

- [ ] **Step 5: Run the authorized local 20-team replay smoke**

Run: `python scripts/discover_kaggle_decks.py discover --max-teams 20 --episodes-per-submission 3`
Expected: checkpointed output, no duplicate episode download, valid 60-card observations, successful replay temp files removed, failures retained and reported. This is the maximum collection scope authorized during implementation.

- [ ] **Step 6: Cluster, report and verify acceptance metrics**

Run: `python scripts/discover_kaggle_decks.py cluster` then `python scripts/discover_kaggle_decks.py report`.
Expected: valid `clusters.jsonl` and `summary.json`; every representative hash exists in `unique_decks.jsonl`; score is absent from clustering features.

- [ ] **Step 7: Commit verification-only fixes, if any**

```powershell
git status --short
git add src/deck_discovery scripts/discover_kaggle_decks.py tests/test_deck_discovery_*.py requirements-train.txt data/external/README.md
git commit -m "test: verify deck discovery smoke workflow"
```

Do not start a server job or the full leaderboard scan. Report smoke counts and ask the user before either action.
