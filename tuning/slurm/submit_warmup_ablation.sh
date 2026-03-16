#!/bin/bash
# ABOUTME: Submits the warmup ratio ablation grid as SLURM jobs.
# ABOUTME: 3 warmup ratios x 2 models x 2 datasets = 12 SFT runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/warmup_ablation.sh"
WANDB_PROJECT="ablations"

WARMUP_RATIOS=(0.0 0.05 0.1)
MODELS=(llama3-3B qwen2-3B)

declare -A TASK_NAME
TASK_NAME["gsm8k"]="gsm8k"
TASK_NAME["tuluif"]="ifeval"

count=0
for dataset in gsm8k tuluif; do
    task="${TASK_NAME[$dataset]}"
    for model in "${MODELS[@]}"; do
        for warmup in "${WARMUP_RATIOS[@]}"; do
            echo "Submitting: dataset=$dataset model=$model warmup_ratio=$warmup"
            sbatch "$SLURM_SCRIPT" \
                --model "$model" \
                --warmup-ratio "$warmup" \
                --dataset "$dataset" \
                --task-name "$task" \
                --train-size 10000 \
                --num-epochs 2 \
                --wandb-project "$WANDB_PROJECT"
            count=$((count + 1))
        done
    done
done

echo ""
echo "Submitted $count jobs. Monitor with: squeue -u \$USER"
