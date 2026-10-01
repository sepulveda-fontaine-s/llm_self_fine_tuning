#!/bin/bash
#SBATCH --job-name=confirm_refined
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:15:00
#SBATCH --output=confirm_refined_%j.out
#SBATCH --error=confirm_refined_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== REFINED CANDIDATE CONFIRMATION ==="

python -u scripts/confirm_refined_candidate.py

echo "=== COMPLETE ==="