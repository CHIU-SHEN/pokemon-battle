#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_DATA="${BASE_DATA:-data/training/training_decisions_v1.jsonl}"
# Only pass a supplement after converting it to training_decision_v1. The
# checked-in exact_supplement_v1.jsonl is observed_decision_v1 and is not a
# valid direct input to AdapterJsonlDataset.
SUPPLEMENT="${SUPPLEMENT:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-artifacts/sl0_shared_full/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/adapters_top10}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-1}"

for view in data/adapter_views/*/view.json; do
  candidate="$(basename "$(dirname "$view")")"
  data_args=(--data "$BASE_DATA")
  if [[ "$candidate" == "alakazam_battle_cage_split" && -n "$SUPPLEMENT" ]]; then
    data_args+=(--data "$SUPPLEMENT")
  fi
  "$PYTHON_BIN" src/train/train_adapter.py \
    --view "$view" \
    "${data_args[@]}" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --output "$OUTPUT_ROOT/$candidate" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device auto
done
