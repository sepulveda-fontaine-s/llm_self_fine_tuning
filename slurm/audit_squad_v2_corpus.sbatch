#!/bin/bash
#SBATCH --job-name=audit_squad_corpus
#SBATCH --output=audit_squad_corpus_%j.out
#SBATCH --error=audit_squad_corpus_%j.err
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

source .venv/bin/activate

echo "=== SQUAD V2 CORPUS AUDIT ==="
python scripts/audit_squad_v2_corpus.py
echo "=== COMPLETE ==="