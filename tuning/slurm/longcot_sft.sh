#!/bin/bash
# ABOUTME: Slurm launcher for long-CoT SFT workers: multi-GPU FSDP2 under torchrun.
# ABOUTME: Banks budget-mark checkpoints; evaluation happens offline on the banked marks.
#SBATCH --job-name=longcot_sft
#SBATCH --partition=gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:4
#SBATCH -c 16
#SBATCH --mem=256GB
#SBATCH --time=3-00:00:00
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

scontrol update JobId="$SLURM_JOB_ID" JobName="$WANDB_PROJECT"

exec > "${SLURM_JOB_ID}_${WANDB_PROJECT}.out" 2>&1

NPROC="${SLURM_GPUS_ON_NODE:-1}"
# Per-job rendezvous port so concurrent workers on a shared node don't collide.
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

echo "[longcot_sft.sh] torchrun --nproc_per_node=${NPROC} --master-port=${MASTER_PORT}"
torchrun --nproc_per_node="${NPROC}" --master-port="${MASTER_PORT}" \
    -m tuning.training.unified_early_pipeline --run-sft --sft-num-gpus "${NPROC}" "$@"
