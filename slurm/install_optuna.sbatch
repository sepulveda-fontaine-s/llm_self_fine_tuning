#!/bin/bash
#SBATCH --job-name=install_optuna
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=install_optuna_%j.out
#SBATCH --error=install_optuna_%j.err

set -euo pipefail

source .venv/bin/activate

pip install -e ".[dev]"

python - <<'PY'
import optuna
print("optuna:", optuna.__version__)
PY

pip check