#!/bin/bash
#SBATCH --job-name=prepare_packed
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=prepare_packed_%j.out
#SBATCH --error=prepare_packed_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== PREPARE PACKED DATASET ==="
python scripts/prepare_packed_dataset.py
echo "=== COMPLETE ==="