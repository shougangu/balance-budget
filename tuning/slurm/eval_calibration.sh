#!/bin/bash
#SBATCH --job-name=eval_calibration
#SBATCH --partition=gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --time=3:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
# The repo must be importable regardless of what the submitting shell exported.
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Concurrent vLLM cold-starts corrupt a shared compile cache; give each job its own root.
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/vllm-cache-${SLURM_JOB_ID}"

ARM_NAME="${ARM_NAME:-arm}"
scontrol update JobId="$SLURM_JOB_ID" JobName="cal-${ARM_NAME}"
exec > "outputs/eval_calibration/${SLURM_JOB_ID}_${ARM_NAME}.out" 2>&1

python scripts/external_eval_calibration.py "$@"
