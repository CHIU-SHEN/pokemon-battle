# MCTS Teacher v2 All-in-One Pipeline Design

## Status

This specification supersedes `2026-08-05-mcts-cpu-gpu-split-design.md` for the current deployment target: one Vast.ai instance with 48 vCPU, 64 GB RAM, and a Tesla V100 32 GB. The dataset archive remains a stable internal boundary, but collection and training ship in one handoff package and run on one server.

## Goal

Build one resumable pipeline that collects 10,000 primary MCTS games on CPU, freezes and verifies the resulting dataset, then trains the MCTS teacher on the V100 and selects a safe checkpoint. The operator should need one archive and one staged job instead of transferring artifacts between CPU and GPU servers.

Reserve data and reserve checkpoints remain outside this iteration.

## Delivery Package

The deliverable is `mcts-teacher-v2-all-in-one.tar.gz`. It contains:

- frozen primary adapter, shared base, and game runtime;
- CPU MCTS collector and worker orchestrator;
- collection audit and immutable dataset archive tools;
- GPU teacher trainer, resume logic, and checkpoint selector;
- smoke, benchmark, collection, training, and full-pipeline entry points;
- verifier, operational documentation, and LF-normalized shell jobs.

The package verifier checks every required member, member SHA-256 values, frozen identity, and the absence of carriage-return bytes in packaged `.sh` files.

## Sequential Architecture

The pipeline runs these stages in order:

1. Environment and identity verification.
2. Ten-game single-worker CPU smoke.
3. Two-hundred-game concurrency benchmark.
4. A 2,400-game safety and throughput gate.
5. Resumable collection up to 10,000 total games.
6. Dataset audit, deduplication, freezing, manifest creation, and hashing.
7. Explicit shutdown and verification of all collection workers.
8. GPU training and holdout evaluation.
9. Safe checkpoint selection and result packaging.

Collection and training never run concurrently. Before GPU training starts, the orchestrator must prove that no managed collector process remains. This avoids CPU, memory, and disk contention and ensures training consumes a fixed dataset.

## CPU Collection

Every worker receives an independent shard directory, seed, iteration ID, log, and atomic `progress.json`. Workers force CPU inference and set:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

The initial worker count is 12. The benchmark evaluates 12, 16, and 20 workers using equal-sized, disjoint game sets. It selects the configuration with the highest completed games per second, provided all safety gates pass and peak memory remains below 80% of available RAM. If a candidate fails, the previous safe candidate remains selected. An explicit worker override is permitted for diagnostics, but the production report records that override.

The benchmark games count toward the 10,000-game target after validation; they are not discarded or recollected. The 2,400-game gate likewise counts toward the final total.

Collection is resumable. A restart rescans completed game files and rebuilds cumulative totals instead of trusting only the last process's incremental counters. Completed, valid game IDs are never scheduled again.

## Collection Safety Gates

The 10-game smoke, each benchmark candidate, the 2,400-game gate, and the final dataset must satisfy:

- `exceptions=0`;
- `illegal_actions=0`;
- `fallback_rate=0` using real action-source accounting;
- every game ID is globally unique;
- every completed shard matches its declared identity and game count;
- train, valid, and test splits are all present;
- total sample count is greater than zero;
- no hidden belief field enters a public training sample;
- primary branch, deck ID, and checkpoint identity match the frozen manifest.

Any failed gate stops progression but preserves all valid completed data for diagnosis and resume.

## Frozen Dataset Boundary

After 10,000 games pass audit, the pipeline creates `mcts-primary-dataset-v2.tar.gz`. Its manifest records schema version, frozen identities, search parameters, seeds, worker benchmark results, shard inventory, game/sample/node totals, safety totals, split counts, deduplication results, source archive provenance, and SHA-256 for every member.

Training accepts only a verified frozen dataset archive. It may merge the authoritative earlier 600-game archive with the new collection, deduplicating by game ID. A stable game-ID hash determines each split so resume and archive order cannot move a game between train, validation, and test.

## GPU Training

Training freezes the shared base and updates the adapter, `policy_delta`, and `value_delta`. It uses visit-count joint-action soft-policy loss, terminal value loss, reference KL regularization, and entropy regularization. Each epoch records policy/value loss, holdout metrics, KL, raw and clipped gradient norm, relative parameter update, its EMA, elapsed time, and current KL coefficient.

The V100 is mandatory for the production training stage. A GPU smoke verifies CUDA allocation, a forward/backward step, checkpoint save, and resume before the full run.

## Adaptive KL and Checkpoints

The initial KL coefficient is `0.05`, with a hard holdout KL limit of `0.03`:

- KL `< 0.015`: multiply the next coefficient by `0.8`, minimum `0.01`;
- KL in `[0.015, 0.025]`: keep the coefficient unchanged;
- KL in `(0.025, 0.03]`: multiply the next coefficient by `2`, maximum `1.0`;
- KL `> 0.03`: stop and retain the preceding safe candidate.

`last.pt` always stores the latest resumable state, including an unsafe final epoch. `best_safe.pt` is replaced only when all values are finite, holdout KL is at most `0.03`, holdout policy loss improves, holdout value loss is no more than 1% worse than the current safe best, and all frozen identities match. Arena evaluation must never consume an unsafe `last.pt`.

After optimizer state is restored, the trainer explicitly reapplies the requested CLI learning rate and the saved adaptive KL coefficient.

## Stopping Rules

Training has no fixed eight-epoch target. It continues until the first applicable condition:

- holdout KL exceeds `0.03`;
- the parameter relative-update EMA remains below the configured convergence threshold for the configured patience after a minimum training duration;
- holdout improvement remains below its configured tolerance for the configured patience after the same minimum duration;
- the 24-hour training time limit is reached;
- a non-finite metric or checkpoint identity failure occurs.

The default minimum training duration is 30 minutes. The default relative-update EMA threshold is `1e-5` with patience 5 epochs. A time-limit stop is reported as `time_limit_reached=true`, not as convergence. The terminal report states the exact stop reason and always points to `best_safe.pt` when one exists.

## Recovery and Time Limits

The collection job has a 24-hour wall-clock limit and the training job has a separate 24-hour limit. Atomic stage state records the active stage, completed shards, selected worker count, dataset hash, training epoch, elapsed time, and stop reason. Re-running the top-level job resumes the incomplete stage after revalidation.

The 10-game smoke has a 30-minute limit. The 2,400-game gate has a six-hour limit. Exceeding a limit preserves state and requires an explicit resume; it does not silently mark the stage successful.

## Outputs

The final `mcts-teacher-v2-results.tar.gz` contains:

- verified dataset manifest and archive hash;
- benchmark and collection summaries;
- training history and stop reason;
- `last.pt` and, when available, `best_safe.pt`;
- checkpoint identity and hashes;
- safety-gate report;
- exact commands and configuration needed for Arena evaluation.

## Testing

Implementation follows TDD and covers worker-candidate selection, shard isolation, resume reconstruction, real fallback counting, duplicate and hidden-field rejection, stage transitions, process shutdown before training, archive hashes, stable splits, adaptive KL boundaries, safe checkpoint promotion, resume parameter reapplication, convergence plus minimum-time behavior, time-limit reporting, CPU smoke, GPU smoke, end-to-end resume, and LF-only packaged shell files.

## Success Criteria

The pipeline succeeds when one verified package can produce a safe, deduplicated 10,000-game primary dataset and a reproducible `best_safe.pt` on the target 48-vCPU/V100 server without manual artifact transfer. Collection safety counters remain zero, training uses only the frozen dataset after collectors stop, unsafe epochs never replace the Arena candidate, and interruption at any stage can resume without duplicating completed games.
