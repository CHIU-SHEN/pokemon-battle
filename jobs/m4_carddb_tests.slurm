#!/bin/bash
#SBATCH --job-name=m4
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

python3 src/cards/card_db.py --out data/cards.json
python3 src/cards/tags.py --cards data/cards.json --overrides data/manual_overrides.json --out data/card_tags.json
python3 tests/test_card_db.py
python3 tests/test_tactics.py
python3 tests/test_regression.py

