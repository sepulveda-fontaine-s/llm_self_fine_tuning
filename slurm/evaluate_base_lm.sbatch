#!/bin/bash
#SBATCH --job-name=base_lm_eval
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=base_lm_eval_%j.out
#SBATCH --error=base_lm_eval_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== BASE LM EVALUATION ==="
python scripts/evaluate_base_lm.py
echo "=== COMPLETE ==="