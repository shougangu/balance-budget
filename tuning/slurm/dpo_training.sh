#!/bin/bash
#SBATCH --job-name=trainingdpo
#SBATCH --partition=gpubase_h100_b3
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=trainingdpo%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

python tuning/training/dpo_training.py