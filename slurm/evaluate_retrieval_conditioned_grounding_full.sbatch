#!/bin/bash
#SBATCH --job-name=rrf_ground_full
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=rrf_ground_full_%j.out
#SBATCH --error=rrf_ground_full_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== RRF GROUNDING / FACTUALITY FULL ==="

python -u scripts/evaluate_retrieval_conditioned_grounding.py \
    --output results/evaluation/retrieval_conditioned_grounding_full.json

echo "=== COMPLETE ==="