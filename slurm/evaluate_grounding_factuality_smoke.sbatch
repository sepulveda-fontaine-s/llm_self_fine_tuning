#!/bin/bash
#SBATCH --job-name=grounding_smoke
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=grounding_smoke_%j.out
#SBATCH --error=grounding_smoke_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== GROUNDING / FACTUALITY SMOKE TEST ==="

python -u scripts/evaluate_grounding_factuality.py \
    --limit 100 \
    --output results/evaluation/grounding_factuality_smoke.json

echo "=== COMPLETE ==="