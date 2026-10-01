#!/bin/bash
#SBATCH --job-name=retrieval_rrf
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=retrieval_rrf_%j.out
#SBATCH --error=retrieval_rrf_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== HYBRID RRF RETRIEVAL BENCHMARK ==="

python -u scripts/evaluate_retrieval_hybrid_rrf.py

echo "=== COMPLETE ==="