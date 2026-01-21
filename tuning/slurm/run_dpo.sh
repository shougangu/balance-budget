#!/bin/bash
#SBATCH --job-name=trainingdpo
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=trainingdpo%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
echo "Running: python tuning/run_dpo.py --model '$1' --dataset '$2' --train_size '$3' --dynamic_path "$4" --sft_train_size '$5' --task '$6' --pft_method '$7' --do_training"

python tuning/run_dpo.py \
  --model "$1" \
  --dataset "$2" \
  --train_size "$3" \
  --dynamic_path "$4" \
  --sft_train_size "$5" \
  --task "$6" \
  --pft_method "$7" \
  --do_training


