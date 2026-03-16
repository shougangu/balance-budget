#!/bin/bash
# ABOUTME: Submits the full beta ablation grid as SLURM jobs.
# ABOUTME: 4 betas x 2 models x 3 checkpoints x 2 datasets = 48 DPO runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_DIR/tuning/models"
SLURM_SCRIPT="$SCRIPT_DIR/beta_ablation.sh"
WANDB_PROJECT="ablations"

BETAS=(0.1 0.3 0.5)

declare -A CHECKPOINTS
# gsm8k checkpoints: early, mid
CHECKPOINTS["gsm8k_llama3-3B"]="llama3-3B_gsm8k-p@1-0.3_sft-1024 llama3-3B_gsm8k-p@1-0.5_sft-7168"
# CHECKPOINTS["gsm8k_qwen2-3B"]="qwen2-3B_gsm8k-p@1-0.65_sft-512 qwen2-3B_gsm8k-p@1-0.75_sft-1536"
# tuluif checkpoints: alternatives used because original llama3-3B (0.35/sft-1024, 0.55/sft-8192)
# and qwen2-3B (0.35/sft-512, 0.6/sft-5632) directories exist but have no model weights.
# llama3-3B: nearest threshold with same or closest sft count that has weights.
# qwen2-2B: only qwen2 family with matching sft counts (512, 5632) and confirmed weights.
CHECKPOINTS["tuluif_llama3-3B"]="llama3-3B_p@1-0.4_sft-1024 llama3-3B_p@1-0.5_sft-3072"
# CHECKPOINTS["tuluif_qwen2-2B"]="qwen2-2B_p@1-0.25_sft-512 qwen2-2B_p@1-0.5_sft-5632"

declare -A TASK_NAME
TASK_NAME["gsm8k"]="gsm8k"
TASK_NAME["tuluif"]="ifeval"

declare -A MODELS
MODELS["gsm8k"]="llama3-3B qwen2-3B"
MODELS["tuluif"]="llama3-3B qwen2-2B"

count=0
for dataset in gsm8k tuluif; do
    task="${TASK_NAME[$dataset]}"
    for model in ${MODELS[$dataset]}; do
        for ckpt in ${CHECKPOINTS[${dataset}_${model}]}; do
            ckpt_path="$MODELS_DIR/$ckpt"
            if [ ! -d "$ckpt_path" ]; then
                echo "WARNING: $ckpt_path does not exist, skipping"
                continue
            fi

            for beta in "${BETAS[@]}"; do
                echo "Submitting: dataset=$dataset model=$model checkpoint=$ckpt beta=$beta"
                sbatch "$SLURM_SCRIPT" \
                    --model "$model" \
                    --checkpoint "$ckpt" \
                    --beta "$beta" \
                    --dataset "$dataset" \
                    --task-name "$task" \
                    --wandb-project "$WANDB_PROJECT"
                count=$((count + 1))
            done
        done
    done
done

echo ""
echo "Submitted $count jobs. Monitor with: squeue -u \$USER"
