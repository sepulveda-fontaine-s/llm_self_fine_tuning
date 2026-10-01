#!/bin/bash
#SBATCH --job-name=retrieval_bm25
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=retrieval_bm25_%j.out
#SBATCH --error=retrieval_bm25_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== BM25 RETRIEVAL BENCHMARK ==="

python -u scripts/evaluate_retrieval_bm25.py

echo "=== COMPLETE ==="