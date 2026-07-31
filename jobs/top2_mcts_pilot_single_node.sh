#!/usr/bin/env bash
set -euo pipefail

: "${PILOT_ROOT:=$PWD/mcts-pilot}"
: "${PILOT_GAMES:=200}"
: "${ARENA_GAMES:=400}"

mkdir -p "$PILOT_ROOT" logs

python scripts/run_top2_mcts_pilot.py \
  --branch primary --output-root "$PILOT_ROOT" \
  --games "$PILOT_GAMES" --arena-games "$ARENA_GAMES" \
  --device cuda --arena-device cpu --resume \
  > logs/mcts-primary.log 2>&1 &
PRIMARY_PID=$!

python scripts/run_top2_mcts_pilot.py \
  --branch reserve --output-root "$PILOT_ROOT" \
  --games "$PILOT_GAMES" --arena-games "$ARENA_GAMES" \
  --device cuda --arena-device cpu --resume \
  > logs/mcts-reserve.log 2>&1 &
RESERVE_PID=$!

wait "$PRIMARY_PID"
wait "$RESERVE_PID"
echo "MCTS PILOT COMPLETE $(date)"
