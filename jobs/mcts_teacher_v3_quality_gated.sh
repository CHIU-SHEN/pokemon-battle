#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-experiments/mcts_teacher_v3/primary}"
WORKERS="${MCTS_WORKERS:-16}"
FINAL_GAMES="${MCTS_GAMES:-5000}"
SIMULATIONS="${MCTS_SIMULATIONS:-128}"
PARTICLES="${MCTS_PARTICLES:-3}"
MAX_DEPTH="${MCTS_MAX_DEPTH:-10}"
TEACHER_GATE_GAMES="${TEACHER_GATE_GAMES:-400}"
TEACHER_MIN_WIN_RATE="${TEACHER_MIN_WIN_RATE:-0.58}"
TIME_BUDGET_SECONDS="${MCTS_TIME_BUDGET_SECONDS:-0.25}"
GAME_BUDGET_SECONDS="${MCTS_GAME_BUDGET_SECONDS:-120}"
DECK_ID="top2-primary-crustle-kangaskhan-cage-v1"
DRY_RUN="${DRY_RUN:-0}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stage() { printf 'STAGE %s\n' "$1"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

stage smoke
run "$PYTHON_BIN" scripts/evaluate_top2_mcts.py \
  --project-root "$ROOT" --branch primary --kind search --games 10 \
  --simulations "$SIMULATIONS" --particles "$PARTICLES" --max-depth "$MAX_DEPTH" \
  --time-budget-seconds "$TIME_BUDGET_SECONDS" --game-budget-seconds "$GAME_BUDGET_SECONDS" \
  --device cpu --output "$RUN_ROOT/smoke.json" --resume

if [[ "${RUN_FULL_PIPELINE:-0}" != "1" ]]; then
  echo "Smoke plan finished. Set RUN_FULL_PIPELINE=1 for the quality-gated run."
  exit 0
fi

stage teacher-evaluation
run "$PYTHON_BIN" scripts/evaluate_top2_mcts.py \
  --project-root "$ROOT" --branch primary --kind search --games "$TEACHER_GATE_GAMES" \
  --simulations "$SIMULATIONS" --particles "$PARTICLES" --max-depth "$MAX_DEPTH" \
  --time-budget-seconds "$TIME_BUDGET_SECONDS" --game-budget-seconds "$GAME_BUDGET_SECONDS" \
  --device cpu --output "$RUN_ROOT/teacher-eval.json" --resume

stage teacher-gate
run "$PYTHON_BIN" scripts/gate_mcts_teacher.py "$RUN_ROOT/teacher-eval.json" \
  --output "$RUN_ROOT/teacher-gate.json" --minimum-games "$TEACHER_GATE_GAMES" \
  --minimum-win-rate "$TEACHER_MIN_WIN_RATE"

stage collection
if [[ "$DRY_RUN" == "1" ]]; then
  for ((worker=0; worker<WORKERS; worker++)); do
    run "$PYTHON_BIN" scripts/collect_top2_mcts.py \
      --project-root "$ROOT" --branch primary --iteration-id "primary-v3-w$(printf '%02d' "$worker")" \
      --games "$((FINAL_GAMES / WORKERS + (worker < FINAL_GAMES % WORKERS ? 1 : 0)))" \
      --simulations "$SIMULATIONS" --particles "$PARTICLES" --max-depth "$MAX_DEPTH" \
      --device cpu --seed "$((20260806 + worker * 1000003))" \
      --time-budget-seconds "$TIME_BUDGET_SECONDS" --game-budget-seconds "$GAME_BUDGET_SECONDS" \
      --output-root "$RUN_ROOT/collection/shards/worker-$(printf '%02d' "$worker")" --resume
  done
else
  mkdir -p "$RUN_ROOT/collection/shards"
  : > "$RUN_ROOT/managed_pids.txt"
  pids=()
  for ((worker=0; worker<WORKERS; worker++)); do
    games=$((FINAL_GAMES / WORKERS + (worker < FINAL_GAMES % WORKERS ? 1 : 0)))
    shard="$RUN_ROOT/collection/shards/worker-$(printf '%02d' "$worker")"
    mkdir -p "$shard"
    "$PYTHON_BIN" scripts/collect_top2_mcts.py \
      --project-root "$ROOT" --branch primary --iteration-id "primary-v3-w$(printf '%02d' "$worker")" \
      --games "$games" --simulations "$SIMULATIONS" --particles "$PARTICLES" \
      --max-depth "$MAX_DEPTH" --device cpu --seed "$((20260806 + worker * 1000003))" \
      --time-budget-seconds "$TIME_BUDGET_SECONDS" --game-budget-seconds "$GAME_BUDGET_SECONDS" \
      --output-root "$shard" --resume > "$shard/collector.log" 2>&1 &
    pids+=("$!")
    echo "$!" >> "$RUN_ROOT/managed_pids.txt"
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  : > "$RUN_ROOT/managed_pids.txt"
  if ((failed)); then echo "At least one collector failed" >&2; exit 1; fi
fi

stage collection-audit
run "$PYTHON_BIN" scripts/audit_mcts_collection.py \
  --root "$RUN_ROOT/collection" --deck-id "$DECK_ID" \
  --expected-games "$FINAL_GAMES" --report "$RUN_ROOT/collection/audit-${FINAL_GAMES}.json"

stage dataset
run "$PYTHON_BIN" scripts/build_mcts_primary_dataset.py \
  --source "$RUN_ROOT/collection" --output "$RUN_ROOT/dataset" --deck-id "$DECK_ID"
run "$PYTHON_BIN" scripts/verify_mcts_primary_dataset.py \
  "$RUN_ROOT/dataset/mcts-primary-dataset-v2.tar.gz"

stage training
resume_args=()
[[ ! -f "$RUN_ROOT/train/last.pt" ]] || resume_args=(--resume "$RUN_ROOT/train/last.pt")
run "$PYTHON_BIN" scripts/train_top2_mcts.py \
  --project-root "$ROOT" --branch primary --samples "$RUN_ROOT/collection" \
  --output "$RUN_ROOT/train" --device cuda --epochs 1000 \
  --learning-rate 0.00001 --kl-coef 0.05 --max-wall-seconds 86400 \
  --min-convergence-seconds 1800 --convergence-patience 5 \
  "${resume_args[@]}"

stage results
if [[ "$DRY_RUN" == "1" ]]; then
  run tar -czf "$RUN_ROOT/mcts-teacher-v3-results.tar.gz" \
    teacher-eval.json teacher-gate.json "collection/audit-${FINAL_GAMES}.json" \
    dataset/mcts-primary-dataset-v2.tar.gz dataset/mcts-primary-dataset-v2.tar.gz.sha256 \
    train/summary.json train/best_safe.pt train/last.pt
  run sha256sum "$RUN_ROOT/mcts-teacher-v3-results.tar.gz"
else
  (
    cd "$RUN_ROOT"
    tar -czf mcts-teacher-v3-results.tar.gz \
      teacher-eval.json teacher-gate.json "collection/audit-${FINAL_GAMES}.json" \
      dataset/mcts-primary-dataset-v2.tar.gz dataset/mcts-primary-dataset-v2.tar.gz.sha256 \
      train/summary.json train/best_safe.pt train/last.pt
    sha256sum mcts-teacher-v3-results.tar.gz > mcts-teacher-v3-results.tar.gz.sha256
  )
fi

echo "Quality-gated pipeline finished. No checkpoint was promoted automatically."
