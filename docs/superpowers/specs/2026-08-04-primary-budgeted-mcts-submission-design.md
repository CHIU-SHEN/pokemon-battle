# Primary Budgeted MCTS Submission Candidate Design

## Goal

Build an isolated, submission-shaped candidate for the validated
`crustle_kangaskhan_cage` primary branch. Preserve the current Abomasnow
`submission/` unchanged until the candidate passes strength, legality, latency,
and packaging gates.

## Candidate policy

The candidate uses the frozen SL-0 shared checkpoint and frozen primary Adapter.
It does not load any failed MCTS-distillation checkpoint. Eligible decisions use
belief-PUCT with an initial budget of eight simulations, one belief particle,
and maximum depth four. Decisions that are ineligible, exceed budget, encounter
an exception, or produce an illegal action fall back to the frozen primary best.

The search loop receives a monotonic deadline and checks it between expansions.
The internal deadline is 30 milliseconds, leaving approximately 5 milliseconds
for observation parsing, legality checks, and fallback under the existing
35-millisecond project budget. The agent also tracks cumulative search time for
the game and stops searching when its configurable per-game allowance is
exhausted.

## Isolation and packaging

Implementation and generated artifacts live outside `submission/`. The package
contains the primary deck, frozen model inputs, lightweight runtime, required
Search API bindings, manifest, and checksums. It must not include raw training
games, reserve-branch assets, failed candidate checkpoints, or development-only
files.

The package exposes conservative environment overrides for simulations,
particles, depth, per-decision budget, and per-game budget while retaining safe
defaults. It never promotes itself or overwrites the formal submission.

## Evaluation gates

Validation proceeds in order:

1. Unit and integration tests for deadline handling, legal fallback, cumulative
   game budget, deterministic configuration, and package completeness.
2. A local smoke run proving zero exceptions and zero illegal actions.
3. A 100-game swapped-seat precheck against the frozen primary best.
4. Only after a positive precheck, a 400-game swapped-seat evaluation.

The candidate is submission-eligible only when the 400-game evaluation has zero
exceptions, zero illegal actions, p95 decision latency no greater than 35
milliseconds, and non-draw win rate at least 55 percent. A missed gate leaves
the current formal submission unchanged and records the failure reason.

## Diagnostics

Reports distinguish searched decisions, policy fallbacks, deadline fallbacks,
game-budget fallbacks, exceptions, and illegal actions. They record mean and p95
latency, nodes per searched decision, total search seconds per game, and the
exact hashes of the deck and frozen checkpoints. This evidence determines
whether to tune budget or abandon the candidate without ambiguity.
