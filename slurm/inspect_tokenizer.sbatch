#!/bin/bash
#SBATCH --job-name=inspect_tokenizer
#SBATCH --output=inspect_tokenizer_%j.out
#SBATCH --error=inspect_tokenizer_%j.err
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

source .venv/bin/activate

echo "=== TOKENIZER INSPECTION ==="
python scripts/inspect_tokenizer.py
echo "=== COMPLETE ==="