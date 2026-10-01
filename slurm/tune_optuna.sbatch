#!/bin/bash
#SBATCH --job-name=optuna_ssl
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=02:00:00
#SBATCH --output=optuna_ssl_%j.out
#SBATCH --error=optuna_ssl_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== OPTUNA SELF-SUPERVISED HYPERPARAMETER SEARCH ==="

python scripts/tune_optuna.py

echo "=== COMPLETE ==="