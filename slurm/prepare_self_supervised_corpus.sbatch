#!/bin/bash
#SBATCH --job-name=prepare_ssl_corpus
#SBATCH --output=prepare_ssl_corpus_%j.out
#SBATCH --error=prepare_ssl_corpus_%j.err
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

source .venv/bin/activate

echo "=== PREPARE SELF-SUPERVISED CORPUS ==="
python scripts/prepare_self_supervised_corpus.py
echo "=== COMPLETE ==="