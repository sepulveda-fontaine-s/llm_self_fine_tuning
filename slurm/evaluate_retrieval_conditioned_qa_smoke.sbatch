#!/bin/bash
#SBATCH --job-name=rrf_qa_smoke
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=rrf_qa_smoke_%j.out
#SBATCH --error=rrf_qa_smoke_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== RETRIEVAL-CONDITIONED QA SMOKE ==="

python -u scripts/evaluate_retrieval_conditioned_qa.py \
    --limit 200 \
    --output results/evaluation/retrieval_conditioned_qa_smoke.json

echo "=== COMPLETE ==="