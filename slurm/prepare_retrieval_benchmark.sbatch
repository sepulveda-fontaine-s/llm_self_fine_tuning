#!/bin/bash
#SBATCH --job-name=prepare_retrieval
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=prepare_retrieval_%j.out
#SBATCH --error=prepare_retrieval_%j.err

set -euo pipefail

source .venv/bin/activate

python -u scripts/prepare_retrieval_benchmark.py