#!/bin/bash
#SBATCH --job-name=evaluate_best_lm
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=evaluate_best_lm_%j.out
#SBATCH --error=evaluate_best_lm_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== BASE VS SELF-SUPERVISED LM EVALUATION ==="
python -u scripts/evaluate_best_lm.py
echo "=== COMPLETE ==="