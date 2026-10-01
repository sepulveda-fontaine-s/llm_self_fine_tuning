#!/bin/bash
#SBATCH --job-name=audit_packing
#SBATCH --output=audit_packing_%j.out
#SBATCH --error=audit_packing_%j.err
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

source .venv/bin/activate

echo "=== PACKING STRATEGY AUDIT ==="
python scripts/audit_packing_strategy.py
echo "=== COMPLETE ==="