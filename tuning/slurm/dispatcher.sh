#!/bin/bash
#SBATCH --job-name=dispatcher
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --gres=gpu:l40s:0
#SBATCH -c 2
#SBATCH --mem=8GB
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

WANDB_PROJECT=""
_args=("$@")
for ((i=0; i<${#_args[@]}; i++)); do
    if [[ "${_args[$i]}" == "--wandb-project" ]] && ((i+1 < ${#_args[@]})); then
        WANDB_PROJECT="${_args[$((i+1))]}"
    fi
done

if [[ -z "$WANDB_PROJECT" ]]; then
    echo "Error: --wandb-project is required" >&2
    exit 1
fi

scontrol update JobId="$SLURM_JOB_ID" JobName="dispatch_${WANDB_PROJECT}"

exec > "${SLURM_JOB_ID}_dispatch_${WANDB_PROJECT}.out" 2>&1

python tuning/training/unified_early_pipeline.py "$@"
