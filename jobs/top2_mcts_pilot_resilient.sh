#!/usr/bin/env bash
set -euo pipefail

: "${PILOT_ROOT:=$PWD/mcts-pilot-200}"
: "${PILOT_GAMES:=200}"
: "${ARENA_GAMES:=400}"

mkdir -p "$PILOT_ROOT" logs

for BRANCH in primary reserve; do
  echo "START branch=$BRANCH $(date --iso-8601=seconds)"
  python scripts/run_top2_mcts_pilot.py \
    --branch "$BRANCH" \
    --output-root "$PILOT_ROOT" \
    --games "$PILOT_GAMES" \
    --arena-games "$ARENA_GAMES" \
    --device cuda \
    --arena-device cpu \
    --resume \
    2>&1 | tee -a "logs/mcts-$BRANCH-resilient.log"
  sync
  echo "COMPLETE branch=$BRANCH $(date --iso-8601=seconds)"
done

echo "MCTS RESILIENT PILOT COMPLETE $(date --iso-8601=seconds)"
