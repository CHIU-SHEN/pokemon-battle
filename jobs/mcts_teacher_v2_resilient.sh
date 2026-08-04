#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-experiments/mcts_teacher_v2/primary}"
ARCHIVE="artifacts/top2-mcts-complete-results-20260804.tar.gz"
EXPANDED="$RUN_ROOT/input"
SAMPLES="$EXPANDED/mcts-scale-600/primary/samples"

mkdir -p "$RUN_ROOT" "$EXPANDED"
echo "f926fbe822d18321d3e083bd30fd60a73da6f35517327f69d0e7bd44262cb531  $ARCHIVE" | sha256sum -c -
if [[ ! -d "$SAMPLES" ]]; then
  tar -xzf "$ARCHIVE" -C "$EXPANDED"
fi

"$PYTHON_BIN" scripts/run_mcts_teacher_smoke.py \
  --project-root "$ROOT" \
  --samples "$SAMPLES" \
  --output "$RUN_ROOT/smoke" \
  --report "$RUN_ROOT/smoke.json"

if [[ "${RUN_TEACHER_TRAIN:-0}" != "1" ]]; then
  echo "Smoke passed. Set RUN_TEACHER_TRAIN=1 to start the bounded primary train."
  exit 0
fi

RESUME_ARGS=()
if [[ -f "$RUN_ROOT/train/last.pt" ]]; then
  RESUME_ARGS=(--resume "$RUN_ROOT/train/last.pt")
fi

"$PYTHON_BIN" scripts/train_top2_mcts.py \
  --project-root "$ROOT" \
  --branch primary \
  --samples "$SAMPLES" \
  --output "$RUN_ROOT/train" \
  --epochs 100 \
  --learning-rate 0.00002 \
  --max-wall-seconds 21600 \
  --checkpoint-interval-seconds 1800 \
  "${RESUME_ARGS[@]}"

echo "Teacher train stopped safely. Arena is intentionally not started automatically."
