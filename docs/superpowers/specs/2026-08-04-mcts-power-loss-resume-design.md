# MCTS Power-Loss Resume Design

## Goal

Make the Top2 MCTS server handoff recover from host power loss without
restarting an in-progress 400-game arena evaluation. The instance disk is
assumed to survive a restart; remote replication is outside this change.

## Runtime design

`evaluate_top2_mcts.py` will accept `--resume`. After every completed game it
will atomically replace a progress JSON file adjacent to the requested output.
The checkpoint records the requested run identity, completed game count,
aggregate results, action-source counts, and decision latencies needed to
produce the existing final report.

On restart, the evaluator will validate that the checkpoint identity matches
the current branch, evaluation kind, target game count, search parameters, and
checkpoint path. A mismatch will fail with a clear error instead of combining
incompatible runs. A valid checkpoint resumes at the next game index, preserving
the alternating first/second-player schedule.

The checkpoint write uses a temporary file in the destination directory,
flushes and fsyncs it, and then replaces the previous checkpoint. This prevents
a power loss during a write from leaving a partially written JSON file.

## Completion behavior

After the target number of games, the evaluator writes the existing formal
result JSON atomically. The progress file remains as audit evidence. A completed
output may be recognized by the orchestration layer and skipped on a later
`--resume` run.

The pilot runner will pass `--resume` to both search and candidate evaluations.
Collection already supports resume, while completed training and evaluation
phases continue to use marker files.

## Server execution

A reliability-oriented single-node script will run `primary` and `reserve`
sequentially. This suits the rented i9-14900KF, 32 GB RAM, and RTX 4060 Ti 8 GB:
arena and MCTS work remain on CPU, and candidate training uses CUDA without two
branches contending for the same GPU.

The script will keep outputs under one configurable persistent directory and
write separate logs for each branch. Re-running the same command after a reboot
will resume incomplete phases.

## Validation

Automated tests will cover:

- interruption after a small number of games followed by an exact resume;
- preservation of first/second-player alternation across restart;
- rejection of incompatible checkpoint metadata;
- recovery when the final output exists but the orchestration marker does not;
- atomic, parseable progress and final JSON output;
- handoff-package verification after rebuilding.

The generated archive and SHA256 sidecar must be rebuilt and verified before
delivery.
