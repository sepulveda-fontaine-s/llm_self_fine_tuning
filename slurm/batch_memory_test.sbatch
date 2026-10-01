#!/bin/bash
#SBATCH --job-name=batch_memory_test
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=00:15:00
#SBATCH --output=batch_memory_test_%j.out
#SBATCH --error=batch_memory_test_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== BATCH MEMORY TEST ==="
python scripts/batch_memory_test.py
echo "=== COMPLETE ==="