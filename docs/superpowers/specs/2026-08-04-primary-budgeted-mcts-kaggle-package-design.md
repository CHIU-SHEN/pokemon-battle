# Primary Budgeted MCTS Kaggle Package Design

## Goal

Produce one Kaggle-uploadable archive for the validated primary
`crustle_kangaskhan_cage` budgeted-MCTS agent, while keeping the existing server
handoff archive and formal Abomasnow submission unchanged.

## Package contract

- Output: `final_submissions/primary_budgeted_mcts_v1.tar.gz`.
- The archive root directly contains `main.py`, `deck.csv`, `cg/`, `agent/`,
  runtime source modules, card data, and frozen primary model assets.
- The archive must not contain a wrapper directory, reserve assets, raw MCTS
  games, training data, or evaluation logs.
- `agent(None)` must return exactly 60 card IDs from the top-level `deck.csv`.
- Runtime defaults remain 8 simulations, 1 particle, depth 4, 30 ms per
  decision, and 2 seconds cumulative search per game.

## Build and verification

Add a dedicated reproducible builder and verifier rather than repurposing the
server handoff builder. Verification checks archive layout, file hashes,
candidate identity, absence of reserve assets, 60-card deck loading, and a
lazy runtime load from a freshly extracted archive.

The release evidence remains separate from the runtime package:

- 400 games: 261 wins, 139 losses, 0 draws.
- Win rate: 65.25%; Wilson lower bound: 60.46%.
- Mean decision latency: 20.96 ms; p95: 30.09 ms.
- Exceptions and illegal actions: 0.
- Full test suite: 86 passed before packaging.

## Documentation

Update `README.md` so the current Kaggle `.tar.gz` requirement supersedes the
older ZIP guidance. Update `项目进度.md` and add a focused release report that
distinguishes the Kaggle archive from server handoff and result archives.

## Safety boundary

Building the package does not upload it to Kaggle, replace `submission/`, or
authorize automatic promotion. Upload remains an explicit user action after
local verification.
