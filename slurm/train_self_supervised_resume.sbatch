#!/bin/bash
#SBATCH --job-name=ssl_train_resume
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:20:00
#SBATCH --output=ssl_train_resume_%j.out
#SBATCH --error=ssl_train_resume_%j.err

set -euo pipefail

export MLFLOW_ALLOW_FILE_STORE=true

source .venv/bin/activate

echo "=== SELF-SUPERVISED TRAINING: RESUME ==="

python -u scripts/train_full.py --resume

echo "=== COMPLETE ==="