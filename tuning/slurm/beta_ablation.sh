#!/bin/bash
# ABOUTME: SLURM job template for a single beta ablation DPO run.
# ABOUTME: Called by submit_beta_ablation.sh with model/checkpoint/beta arguments.
#SBATCH --job-name=beta_ablation
#SBATCH --partition=gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=80GB
#SBATCH --time=3:00:00
#SBATCH --output=beta_ablation_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

python tuning/training/beta_ablation.py "$@"
