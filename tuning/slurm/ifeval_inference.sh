#!/bin/bash
#SBATCH --job-name=if1000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=greedy_hf_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

python tuning/inference/ifeval_inference.py