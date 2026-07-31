#!/usr/bin/env bash
#SBATCH --job-name=top2-sp-train
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/selfplay-train.%A_%a.out
#SBATCH --error=logs/selfplay-train.%A_%a.err

set -euo pipefail
BRANCHES=(primary reserve)
BRANCH="${BRANCHES[$SLURM_ARRAY_TASK_ID]}"
: "${ITERATION_ID:?set ITERATION_ID}"
: "${SELFPLAY_ROOT:=selfplay}"

python scripts/run_top2_selfplay_iteration.py \
  --branch "$BRANCH" \
  --selfplay-root "$SELFPLAY_ROOT" \
  --iteration-id "$ITERATION_ID" \
  --stop-after train \
  --resume \
  --device cuda
