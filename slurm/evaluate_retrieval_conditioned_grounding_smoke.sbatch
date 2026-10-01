#!/bin/bash
#SBATCH --job-name=rrf_ground_smoke
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=rrf_ground_smoke_%j.out
#SBATCH --error=rrf_ground_smoke_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== RRF GROUNDING SMOKE ==="

python -u scripts/evaluate_retrieval_conditioned_grounding.py \
    --limit 200 \
    --output results/evaluation/retrieval_conditioned_grounding_smoke.json

echo "=== COMPLETE ==="