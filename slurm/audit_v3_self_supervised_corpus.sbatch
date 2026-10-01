#!/bin/bash
#SBATCH --job-name=audit_v3_ssl
#SBATCH --output=audit_v3_ssl_%j.out
#SBATCH --error=audit_v3_ssl_%j.err
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

source .venv/bin/activate

echo "=== V3 SELF-SUPERVISED CORPUS AUDIT ==="
python scripts/audit_v3_self_supervised_corpus.py
echo "=== COMPLETE ==="