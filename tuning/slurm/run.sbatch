#!/bin/bash
#
# Available partitions: COMPLING or H200
# SBATCH settings
#SBATCH --job-name=runsh
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=test_python_%j.out

# Make sure to run export PYTHONPATH="$(pwd)" in balance-budget repo
# Example Usage: sbatch --export=ALL,TASK=instruction,DATASET=tuluif,DO_TRAINING=y,DO_INFERENCE=y,DO_EVALUATION=y,DO_SFT_FIRST=y run.sbatch
# Check status of job:
# watch -n 5 squeue -u $USER
# scontrol show job ID

# Recommended safety flags
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

# --- Defaults here (overridable via --export=ALL,VAR=... ) ---
export TASK="${TASK:-instruction}"
export DATASET="${DATASET:-tuluif}"
export SFT_RATIO="${SFT_RATIO:-0.75}"
export BASE_MODEL="${BASE_MODEL:-llama3-8B}"
export PFT_METHOD="${PFT_METHOD:-dpo}"

export DO_TRAINING="${DO_TRAINING:-y}"
export DO_INFERENCE="${DO_INFERENCE:-y}"
export DO_EVALUATION="${DO_EVALUATION:-y}"
export DO_SFT_FIRST="${DO_SFT_FIRST:-y}"

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

# export CONDA_ENV="${CONDA_ENV:-bb-train}"

# --- Conda activation (batch-safe) ---
# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/anaconda3/etc/profile.d/conda.sh"
# else
#   CONDA_BASE="$(conda info --base 2>/dev/null || true)"
#   [ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
# fi
# conda activate "$CONDA_ENV"
# echo "Activating conda environment: $CONDA_ENV"

# Optional: log what we're about to run
echo "Running with: TASK=$TASK DATASET=$DATASET SFT_RATIO=$SFT_RATIO BASE_MODEL=$BASE_MODEL PFT_METHOD=$PFT_METHOD"
echo "Toggles: DO_TRAINING=$DO_TRAINING DO_INFERENCE=$DO_INFERENCE DO_EVALUATION=$DO_EVALUATION DO_SFT_FIRST=$DO_SFT_FIRST"

# Just run it (no stdin piping needed now)
srun --export=ALL /bin/bash tuning/run.sh
