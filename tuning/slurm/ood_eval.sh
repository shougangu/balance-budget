#!/bin/bash
# ABOUTME: Slurm wrapper for scripts/ood_eval.py — OOD benchmark eval of budget-grid
# ABOUTME: target snapshots on one GPU; all script args pass through.
#SBATCH --job-name=ood_eval
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=128GB
#SBATCH --time=12:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
# Concurrent vLLM cold-starts corrupt a shared compile cache ("Bytes object
# is corrupted"); give each job its own node-local cache root.
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/vllm-cache-${SLURM_JOB_ID}"

exec .venv/bin/python scripts/ood_eval.py "$@"
