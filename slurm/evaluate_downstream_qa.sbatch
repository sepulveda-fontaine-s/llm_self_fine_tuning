#!/bin/bash
#SBATCH --job-name=ssl_downstream_qa
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=01:00:00
#SBATCH --output=ssl_downstream_qa_%j.out
#SBATCH --error=ssl_downstream_qa_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== DOWNSTREAM QA EVALUATION ==="

python -u scripts/evaluate_downstream_qa.py

echo "=== COMPLETE ==="