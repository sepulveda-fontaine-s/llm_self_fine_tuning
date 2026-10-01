#!/bin/bash
#SBATCH --job-name=ssl_train_refined
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:20:00
#SBATCH --output=ssl_train_refined_%j.out
#SBATCH --error=ssl_train_refined_%j.err

set -euo pipefail

export MLFLOW_ALLOW_FILE_STORE=true

source .venv/bin/activate

echo "=== FINAL REFINED SELF-SUPERVISED TRAINING ==="

python -u scripts/train_full.py

echo "=== COMPLETE ==="