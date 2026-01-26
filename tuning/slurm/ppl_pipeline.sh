#!/bin/bash
#SBATCH --job-name=ppl_pipeline
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=ppl_pipeline%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

python tuning/training/ppl_pipeline.py