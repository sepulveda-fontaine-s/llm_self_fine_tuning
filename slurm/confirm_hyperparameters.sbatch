#!/bin/bash
#SBATCH --job-name=confirm_ssl_hparams
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=01:30:00
#SBATCH --output=confirm_ssl_hparams_%j.out
#SBATCH --error=confirm_ssl_hparams_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== HYPERPARAMETER CONFIRMATION ==="
python scripts/confirm_hyperparameters.py
echo "=== COMPLETE ==="