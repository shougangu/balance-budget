#!/bin/bash
# ABOUTME: Slurm launcher for one verl GRPO worker: claims a banked SFT mark and spends
# ABOUTME: the remainder of its cell budget. Single node, ray local mode, verl venv.
#SBATCH --job-name=verl_grpo
#SBATCH --gres=gpu:h100:4
#SBATCH -c 16
#SBATCH --mem=256GB
#SBATCH --time=1-00:00:00
#SBATCH --output=%j_verl_grpo.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/vllm-cache-${SLURM_JOB_ID}"
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray-${SLURM_JOB_ID}"
# Slurm exports the AMD device list next to CUDA_VISIBLE_DEVICES; ray refuses
# to start GPU workers when both are set.
unset ROCR_VISIBLE_DEVICES

# The computecanada wheels lean on cluster modules at runtime: cv2 from opencv,
# pyarrow from arrow (see tuning/verl/SETUP.md). The arrow version is pinned to
# the one the venv's pyarrow placeholder dist was registered against.
module load gcc arrow/19.0.1 opencv

# The verl venv lives beside the repo clone on every cluster.
VERL_VENV="${VERL_VENV:-$(dirname "$SLURM_SUBMIT_DIR")/venvs/verl-0.9.0}"

"${VERL_VENV}/bin/python" -m tuning.verl.run_verl_grpo "$@"
