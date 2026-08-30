#!/bin/bash
#SBATCH --job-name=if_grid_eval
#SBATCH --partition=gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
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

CELL_NAME="${CELL_NAME:-cell}"
scontrol update JobId="$SLURM_JOB_ID" JobName="if-${CELL_NAME}"
mkdir -p outputs/if_grid_n48
exec > "outputs/if_grid_n48/${SLURM_JOB_ID}_${CELL_NAME}.out" 2>&1

# A 12B merge is ~24 GB and this cell is its only reader, so it does not outlive the job.
merge_root=""
for ((i = 1; i <= $#; i++)); do
    if [[ "${!i}" == "--merge-root" ]]; then
        j=$((i + 1))
        merge_root="${!j}"
    fi
done
cleanup() {
    if [[ -n "$merge_root" && "$merge_root" == /scratch/* ]]; then
        rm -rf "$merge_root"
    fi
}
trap cleanup EXIT

python scripts/external_eval_calibration.py "$@"
