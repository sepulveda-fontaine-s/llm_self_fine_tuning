#!/bin/bash
#SBATCH --job-name=optuna_ssl_refined
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=03:00:00
#SBATCH --output=optuna_ssl_refined_%j.out
#SBATCH --error=optuna_ssl_refined_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== REFINED OPTUNA SELF-SUPERVISED SEARCH ==="

python -u scripts/tune_optuna.py \
    --tuning-section tuning_refined

echo "=== COMPLETE ==="