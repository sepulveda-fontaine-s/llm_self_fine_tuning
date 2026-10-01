#!/bin/bash
#SBATCH --job-name=ssl_smoke_test
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:15:00
#SBATCH --output=ssl_smoke_test_%j.out
#SBATCH --error=ssl_smoke_test_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== SELF-SUPERVISED TRAINING SMOKE TEST ==="
python scripts/model_training_smoke_test.py
echo "=== COMPLETE ==="