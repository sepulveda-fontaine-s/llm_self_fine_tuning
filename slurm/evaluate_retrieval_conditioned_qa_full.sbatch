#!/bin/bash
#SBATCH --job-name=rrf_qa_full
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=rrf_qa_full_%j.out
#SBATCH --error=rrf_qa_full_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== RETRIEVAL-CONDITIONED QA FULL ==="

python -u scripts/evaluate_retrieval_conditioned_qa.py \
    --output results/evaluation/retrieval_conditioned_qa_refined_full.json

echo "=== COMPLETE ==="