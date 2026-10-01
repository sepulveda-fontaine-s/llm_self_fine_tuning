#!/bin/bash
#SBATCH --job-name=ssl_train_epoch1
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:30:00
#SBATCH --output=ssl_train_epoch1_%j.out
#SBATCH --error=ssl_train_epoch1_%j.err

set -euo pipefail

export MLFLOW_ALLOW_FILE_STORE=true

source .venv/bin/activate

echo "=== SELF-SUPERVISED TRAINING: EPOCH 1 ==="

python -u scripts/train_full.py \
    --max-epochs-this-run 1

echo "=== COMPLETE ==="