#!/bin/bash
#SBATCH --job-name=logP500test
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --output=greedy_hf_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

for model in llama3-8B_sft-tuluif-5000 llama3-8B_sft-tuluif-10000
do
    python tuning/run_perplexity_check.py --model_name "$model"
done
