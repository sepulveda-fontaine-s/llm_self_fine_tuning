#!/bin/bash
#SBATCH --job-name=deploy_index
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=deploy_index_%j.out
#SBATCH --error=deploy_index_%j.err

set -euo pipefail

source .venv/bin/activate

echo "=== BUILD DEPLOYMENT RETRIEVAL INDEX ==="

python -u scripts/build_deployment_retrieval_index.py

echo "=== COMPLETE ==="