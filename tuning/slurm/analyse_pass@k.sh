#!/bin/bash
#SBATCH --job-name=pass_at_k_eval
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --output=pass_at_k_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1


# Run pass@k evaluation for multiple models
# for model in llama3-8B_sft-tuluif-250 llama3-8B_sft-tuluif-5000 llama3-8B_sft-tuluif-10000
# model="llama3-8B" 
model="llama3-8B_sft-tuluif-250"
# model="llama3-8B_sft-tuluif-250_pt-tuluif-750"
# model="llama3-8B_sft-tuluif-500"
# model="llama3-8B_sft-tuluif-500_pt-tuluif-500"
# model="llama3-8B_sft-tuluif-750"
# model="llama3-8B_sft-tuluif-750_pt-tuluif-250"
# model="llama3-8B_sft-tuluif-1000" 
# model="llama3-8B_sft-tuluif-5000"
# model="llama3-8B_sft-tuluif-5000_pt-tuluif-5000"
# model="llama3-8B_sft-tuluif-10000"
# do
    python tuning/pass@k.py \
        --model "$model" \
        --skip-inference True \ 
        --k-values 1 2 4 8 16 32 64\
        --n-samples 128 \
        --temperature 0.7
# done