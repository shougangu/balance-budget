#!/bin/bash
#SBATCH --job-name=unified_early_pipeline
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=unified_early_pipeline_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

python tuning/training/unified_early_pipeline.py "$@"
