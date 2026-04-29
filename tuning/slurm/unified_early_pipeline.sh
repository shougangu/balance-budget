#!/bin/bash
#SBATCH --job-name=unified_early_pipeline
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

# Extract --wandb-project, --task-name, and --grpo-num-gpus from arguments
WANDB_PROJECT=""
TASK_NAME="gsm8k"
_args=("$@")
for ((i=0; i<${#_args[@]}; i++)); do
    if [[ "${_args[$i]}" == "--wandb-project" ]] && ((i+1 < ${#_args[@]})); then
        WANDB_PROJECT="${_args[$((i+1))]}"
    elif [[ "${_args[$i]}" == "--task-name" ]] && ((i+1 < ${#_args[@]})); then
        TASK_NAME="${_args[$((i+1))]}"
    fi
done

if [[ -z "$WANDB_PROJECT" ]]; then
    echo "Error: --wandb-project is required" >&2
    exit 1
fi

scontrol update JobId="$SLURM_JOB_ID" JobName="$WANDB_PROJECT"

exec > "${SLURM_JOB_ID}_${WANDB_PROJECT}.out" 2>&1

# GRPO worker mode launches under torchrun for DDP across all node GPUs.
# Other modes (orchestrator, --run-sft, --run-dpo) use plain python.
IS_GRPO_WORKER=0
for _arg in "$@"; do
    if [[ "$_arg" == "--run-grpo" ]]; then
        IS_GRPO_WORKER=1
        break
    fi
done

NPROC="${SLURM_GPUS_ON_NODE:-1}"


# Per-job rendezvous port so concurrent GRPO workers on the same node
# don't collide on torchrun's default port 29500.
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

if [[ "$IS_GRPO_WORKER" == "1" ]]; then
    echo "[unified_early_pipeline.sh] GRPO worker mode: torchrun --nproc_per_node=${NPROC} --master-port=${MASTER_PORT}"
    torchrun --nproc_per_node="${NPROC}" --master-port="${MASTER_PORT}" \
        -m tuning.training.unified_early_pipeline "$@"
else
    python tuning/training/unified_early_pipeline.py "$@"
fi

# Generate sweetspot table from the W&B project
WANDB_ENTITY=$(python -c "import wandb; print(wandb.Api().default_entity)")
SWEETSPOT_FLAGS=()
if [[ "$TASK_NAME" == "ifeval" ]]; then
    SWEETSPOT_FLAGS+=(--ifeval)
fi
python scripts/sweetspot_table.py "https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}" "${SWEETSPOT_FLAGS[@]}"
