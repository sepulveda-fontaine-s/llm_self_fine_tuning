#!/bin/bash
#SBATCH --job-name=retrieval_dense
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=retrieval_dense_%j.out
#SBATCH --error=retrieval_dense_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== DENSE RETRIEVAL BENCHMARK ==="

python -u scripts/evaluate_retrieval_dense.py

echo "=== COMPLETE ==="