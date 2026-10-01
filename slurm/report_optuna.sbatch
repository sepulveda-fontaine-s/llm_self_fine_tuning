#!/bin/bash
#SBATCH --job-name=report_optuna
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=report_optuna_%j.out
#SBATCH --error=report_optuna_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== OPTUNA REPORT ==="
python scripts/report_optuna.py
echo "=== COMPLETE ==="