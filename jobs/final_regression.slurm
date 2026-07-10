#!/bin/bash
#SBATCH --job-name=final
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

mkdir -p logs

python3 tests/test_regression.py
python3 tests/test_fallback.py
python3 tests/test_m3_search.py
python3 tests/test_card_db.py
python3 tests/test_tactics.py
python3 tests/test_deck_optimizer.py
python3 tests/test_distill_pipeline.py
python3 tests/test_flat_submission.py

python3 scripts/freeze_final.py

python3 eval/run_match.py \
  --agent0 final_submissions/submission_flat_safe_v0/main.py \
  --agent1 random \
  --games 1000 \
  --seed 20260707 \
  --out-dir experiments/final_safe_vs_random_1000

python3 scripts/freeze_final.py
