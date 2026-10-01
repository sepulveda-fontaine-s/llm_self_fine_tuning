#!/bin/bash
#SBATCH --job-name=grounding_full
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=grounding_full_%j.out
#SBATCH --error=grounding_full_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== FULL GROUNDING / FACTUALITY EVALUATION ==="

python -u scripts/evaluate_grounding_factuality.py \
    --output results/evaluation/grounding_factuality_refined_full.json

echo "=== COMPLETE ==="