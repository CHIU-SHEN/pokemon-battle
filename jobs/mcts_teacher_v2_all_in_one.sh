#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-experiments/mcts_teacher_v2_all_in_one/primary}"
WORKERS="${MCTS_WORKERS:-16}"
FINAL_GAMES="${MCTS_GAMES:-10000}"
GATE_GAMES="${MCTS_GATE_GAMES:-2400}"
DECK_ID="top2-primary-crustle-kangaskhan-cage-v1"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RUN_ROOT"

run_worker_stage() {
  local target="$1"
  local base=$((target / WORKERS))
  local remainder=$((target % WORKERS))
  local pids=()
  : > "$RUN_ROOT/managed_pids.txt"
  for ((worker=0; worker<WORKERS; worker++)); do
    local games="$base"
    if ((worker < remainder)); then games=$((games + 1)); fi
    local shard="$RUN_ROOT/collection/shards/worker-$(printf '%02d' "$worker")"
    mkdir -p "$shard"
    "$PYTHON_BIN" scripts/collect_top2_mcts.py \
      --project-root "$ROOT" --branch primary \
      --iteration-id "primary-10k-w$(printf '%02d' "$worker")" \
      --games "$games" --device cpu --seed "$((20260805 + worker * 1000003))" \
      --output-root "$shard" --resume \
      > "$shard/collector.log" 2>&1 &
    pids+=("$!")
    echo "$!" >> "$RUN_ROOT/managed_pids.txt"
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  : > "$RUN_ROOT/managed_pids.txt"
  if ((failed)); then echo "At least one collector failed" >&2; return 1; fi
  "$PYTHON_BIN" scripts/audit_mcts_collection.py \
    --root "$RUN_ROOT/collection" --deck-id "$DECK_ID" \
    --expected-games "$target" --report "$RUN_ROOT/collection/audit-${target}.json"
}

if [[ ! -f "$RUN_ROOT/smoke.complete" ]]; then
  "$PYTHON_BIN" scripts/collect_top2_mcts.py \
    --project-root "$ROOT" --branch primary --iteration-id primary-smoke \
    --games 10 --device cpu --output-root "$RUN_ROOT/smoke"
  touch "$RUN_ROOT/smoke.complete"
fi

if [[ "${RUN_FULL_PIPELINE:-0}" != "1" ]]; then
  echo "CPU smoke passed. Set RUN_FULL_PIPELINE=1 to collect, freeze, and train."
  exit 0
fi

if [[ ! -f "$RUN_ROOT/collection/audit-${GATE_GAMES}.json" && ! -f "$RUN_ROOT/collection/audit-${FINAL_GAMES}.json" ]]; then
  run_worker_stage "$GATE_GAMES"
fi
if [[ ! -f "$RUN_ROOT/collection/audit-${FINAL_GAMES}.json" ]]; then
  run_worker_stage "$FINAL_GAMES"
fi

while read -r pid; do
  [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null || { echo "collector still running: $pid" >&2; exit 1; }
done < "$RUN_ROOT/managed_pids.txt"

"$PYTHON_BIN" scripts/build_mcts_primary_dataset.py \
  --source "$RUN_ROOT/collection" --output "$RUN_ROOT/dataset" --deck-id "$DECK_ID"
"$PYTHON_BIN" scripts/verify_mcts_primary_dataset.py "$RUN_ROOT/dataset/mcts-primary-dataset-v2.tar.gz"

RESUME_ARGS=()
[[ ! -f "$RUN_ROOT/train/last.pt" ]] || RESUME_ARGS=(--resume "$RUN_ROOT/train/last.pt")
"$PYTHON_BIN" scripts/train_top2_mcts.py \
  --project-root "$ROOT" --branch primary --samples "$RUN_ROOT/collection" \
  --output "$RUN_ROOT/train" --device cuda --epochs 1000 \
  --learning-rate 0.00001 --kl-coef 0.05 --max-wall-seconds 86400 \
  --min-convergence-seconds 1800 --convergence-patience 5 \
  "${RESUME_ARGS[@]}"

echo "Pipeline finished. Arena candidate: $RUN_ROOT/train/best_safe.pt"
