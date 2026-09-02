#!/bin/bash
# ABOUTME: Slurm wrapper for the offline checkpoint eval: vLLM inference on banked marks.
# ABOUTME: GPU count/time are set by the submitter (--gres/--time on the sbatch line).
#SBATCH --job-name=offline_eval
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=%j_offline_eval.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/vllm-cache-${SLURM_JOB_ID}"

.venv/bin/python scripts/offline_checkpoint_eval.py "$@"
