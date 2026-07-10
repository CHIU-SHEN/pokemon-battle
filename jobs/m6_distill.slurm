#!/bin/bash
#SBATCH --job-name=m6
#SBATCH -D /share/home/shenhanxi
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --partition=testLM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=10245000454@stu.ecnu.edu.cn

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/home/shenhanxi/pokemon}"
cd "$PROJECT_DIR"

source jobs/slurm_env.sh
activate_pokemon_env

mkdir -p logs data/distill models

python3 src/train/collect_distill.py \
  --fixtures tests/fixtures/observations.json \
  --out data/distill/smoke_samples.jsonl \
  --schema-out data/distill/schema.json \
  --summary-out data/distill/collect_summary.json \
  --max-samples 50 \
  --group-size 10 \
  --search \
  --max-candidates 4 \
  --particles 1 \
  --node-budget 16 \
  --time-budget 0.02

python3 src/train/train_distill.py \
  --data data/distill/smoke_samples.jsonl \
  --model-out models/v2_policy_linear.json \
  --metrics-out data/distill/train_metrics.json \
  --splits-out data/distill/splits.json \
  --epochs 80

python3 src/train/reanalysis.py \
  --data data/distill/smoke_samples.jsonl \
  --model models/v2_policy_linear.json \
  --out data/reanalysis_queue.json \
  --max-items 100

python3 tests/test_distill_pipeline.py

