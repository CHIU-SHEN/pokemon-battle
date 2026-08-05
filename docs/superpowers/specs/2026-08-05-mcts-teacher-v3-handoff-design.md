# MCTS Teacher v3 Handoff Design

## Objective

Produce one checksum-verified server handoff that tests whether the search
teacher is strong enough before spending compute on a new distillation run.
If the teacher passes, the same resumable job collects a smaller but stronger
dataset and trains the next candidate.

## Selected approach

Extend the existing all-in-one handoff and job rather than create unrelated
manual commands or separate CPU/GPU archives. A single package preserves the
frozen model inputs, project code, audit scripts, job defaults, and resumable
state contract.

Two alternatives were rejected:

- Separate teacher-gate and training packages reduce archive size but create
  an avoidable handoff boundary and checksum/version mismatch risk.
- A command-only runbook is easy to create but does not provide reproducible
  defaults, automatic stopping, or package verification.

## Pipeline

1. Run a 10-game smoke test.
2. Evaluate the direct MCTS search teacher for 400 swapped-seat games against
   the current deterministic baseline.
3. Stop unless decisive-game win rate is at least 58%, with zero exceptions,
   zero illegal actions, and zero MCTS safety fallbacks.
4. On pass, collect 5,000 games using 128 simulations, 3 particles, maximum
   depth 10, balanced seats, CPU workers, resumable shards, and no root-policy
   promotion.
5. Audit the collection and fail on any exception, illegal action, or safety
   fallback. Record policy fallback separately as a data-quality statistic.
6. Freeze and verify the dataset.
7. Train on CUDA using the existing convergence, holdout, wall-time, and
   best-safe-checkpoint controls.
8. Package teacher evaluation, dataset audit, dataset archive, training
   summary, `best_safe.pt`, and checksums as downloadable outputs.

## Configuration and overrides

Safe defaults are encoded in the job:

- `TEACHER_GATE_GAMES=400`
- `TEACHER_MIN_WIN_RATE=0.58`
- `MCTS_GAMES=5000`
- `MCTS_SIMULATIONS=128`
- `MCTS_PARTICLES=3`
- `MCTS_MAX_DEPTH=10`
- `MCTS_WORKERS=16`

Worker count and wall-time budgets remain environment overrides. Search
quality defaults must be explicit in every collector invocation so a server
cannot silently fall back to the old 32/8 settings.

## Opponent behavior

The teacher gate uses the current deterministic deployed baseline so its
result remains comparable with earlier Arena reports. The new collection
records opponent identity and is structured to accept a weighted opponent
schedule in a later DAgger iteration. This package does not synthesize a
multi-opponent pool without validated compatible checkpoints; doing so would
make the first teacher-strength diagnosis ambiguous.

## Failure and resume behavior

Every expensive phase writes progress before continuing. Re-running the job
with the same run root resumes teacher evaluation, worker shards, dataset
creation, or training. A failed teacher gate exits nonzero after preserving
its report and does not collect or train. Existing output with an incompatible
identity or search configuration is rejected rather than reused.

## Tests

- CLI/config tests prove 128 simulations and depth 10 reach search and
  collection code.
- Gate tests cover pass, weak-teacher, and unsafe-teacher outcomes.
- Job/package tests verify stage ordering, resumable flags, LF shell files,
  manifest values, checksums, and required frozen inputs.
- A handoff smoke verification extracts the archive and validates every member.

## Deliverables

- `mcts-teacher-v3-quality-gated.tar.gz`
- `mcts-teacher-v3-quality-gated.tar.gz.sha256`
- Server runbook contained in the archive
- Local implementation and test evidence

The package never automatically promotes a model. Arena evaluation and an
explicit promotion decision remain separate after the result is downloaded.

## Post-training hybrid evaluation decision

The distilled policy is not required to be the only deployment form. After
the v3 result returns, evaluate the same policy/value checkpoint in four
matched configurations: pure policy (0 simulations), policy-guided MCTS with
8 simulations, policy-guided MCTS with 16 simulations, and the 128-simulation
teacher. Use identical swapped-seat seeds and report both Arena strength and
decision latency.

This follows the useful part of the Kaggle AlphaZero-style sample: the network
provides priors and leaf values while a small online search remains available
to correct distribution-shift errors. The 53% promotion target remains, but
selection is now a strength-latency decision. Prefer the lowest-search option
that passes all safety gates and reaches the target. See
`docs/MCTS_HYBRID_POLICY_DECISION.md` for the authoritative evaluation matrix.
