#!/bin/bash

train_sizes=( 1000 ) #1000 10000 )

# Set boolean config values to "n" by default
export DO_TRAINING="${DO_TRAINING:-n}"
export DO_INFERENCE="${DO_INFERENCE:-n}"
export DO_EVALUATION="${DO_EVALUATION:-n}"
export DO_SFT_FIRST="${DO_SFT_FIRST:-n}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) export TASK="$2"; shift 2 ;;
        --dataset) export DATASET="$2"; shift 2 ;;
        --sft_ratio) export SFT_RATIO="$2"; shift 2 ;;
        --base_model) export BASE_MODEL="$2"; shift 2 ;;
        --pft_method) export PFT_METHOD="$2"; shift 2 ;;
        --do_training) export DO_TRAINING="y"; shift ;;
        --do_inference) export DO_INFERENCE="y"; shift ;;
        --do_evaluation) export DO_EVALUATION="y"; shift ;;
        --do_sft_first) export DO_SFT_FIRST="y"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

set -e

# Determine config source for echo
if [[ -n "${TASK:-}" && -n "${DATASET:-}" && -n "${SFT_RATIO:-}" && -n "${BASE_MODEL:-}" && -n "${PFT_METHOD:-}" ]]; then
    if [[ "$*" != "" ]]; then
        config_source="Command-line arguments (overriding env vars)"
    else
        config_source="Environment variables"
    fi
else
    config_source="Interactive mode (prompts)"
fi

echo "Configuration source: $config_source"
echo "  TASK=${TASK:-}"
echo "  DATASET=${DATASET:-}"
echo "  SFT_RATIO=${SFT_RATIO:-}"
echo "  BASE_MODEL=${BASE_MODEL:-}"
echo "  PFT_METHOD=${PFT_METHOD:-}"
echo "  DO_TRAINING=${DO_TRAINING:-}"
echo "  DO_INFERENCE=${DO_INFERENCE:-}"
echo "  DO_EVALUATION=${DO_EVALUATION:-}"
echo "  DO_SFT_FIRST=${DO_SFT_FIRST:-}"

# --- Non-interactive fast path (use env vars if provided) ---
if [[ -n "${TASK:-}" || -n "${DATASET:-}" || -n "${SFT_RATIO:-}" || -n "${BASE_MODEL:-}" || -n "${PFT_METHOD:-}" ]]; then
  # Check for missing required variables
  missing_vars=()
  [[ -z "${TASK:-}" ]] && missing_vars+=("TASK")
  [[ -z "${DATASET:-}" ]] && missing_vars+=("DATASET")
  [[ -z "${SFT_RATIO:-}" ]] && missing_vars+=("SFT_RATIO")
  [[ -z "${BASE_MODEL:-}" ]] && missing_vars+=("BASE_MODEL")
  [[ -z "${PFT_METHOD:-}" ]] && missing_vars+=("PFT_METHOD")
  if (( ${#missing_vars[@]} > 0 )); then
    echo "Error: Missing required config values: ${missing_vars[*]}"
    echo "You must specify all required flags or environment variables."
    exit 1
  fi

  echo "Mode: Non-interactive (using command-line arguments or environment variables)"
  task="$TASK"
  dataset="$DATASET"
  sft_ratio="$SFT_RATIO"
  base_model="$BASE_MODEL"
  pft_method="$PFT_METHOD"

  # yes/no toggles from env (default to n)
  do_training="${DO_TRAINING:-n}"
  do_inference="${DO_INFERENCE:-n}"
  do_evaluation="${DO_EVALUATION:-n}"
  do_sft_first="${DO_SFT_FIRST:-n}"

  training_flag=$([ "${do_training,,}" = "y" ] && echo "--do_training" || echo "")
  inference_flag=$([ "${do_inference,,}" = "y" ] && echo "--do_inference" || echo "")
  evaluation_flag=$([ "${do_evaluation,,}" = "y" ] && echo "--do_evaluation" || echo "")

  for train_size in "${train_sizes[@]}"; do
      sft_train_size=$(awk "BEGIN {print $train_size * $sft_ratio}")
      dpo_train_size=$(awk "BEGIN {print $train_size * (1 - $sft_ratio)}")

      if [ "$sft_ratio" = "1.0" ]; then
          python tuning/run_sft.py \
              --train_size "$sft_train_size" \
              --model "$base_model" \
              --dataset "$dataset" \
              --task "$task" \
              $training_flag \
              $inference_flag \
              $evaluation_flag
      else
          if [ "${do_sft_first,,}" = "y" ]; then
              echo "Running SFT first (training only)..."
              python tuning/run_sft.py \
                  --train_size "$sft_train_size" \
                  --model "$base_model" \
                  --dataset "$dataset" \
                  --task "$task" \
                  --do_training
          fi

          python tuning/run_dpo.py \
              --train_size "$dpo_train_size" \
              --model "$base_model" \
              --dataset "$dataset" \
              --sft_train_size "$sft_train_size" \
              --task "$task" \
              --pft_method "$pft_method" \
              $training_flag \
              $inference_flag \
              $evaluation_flag
      fi
  done
  exit 0
else
  echo "Mode: Interactive (using prompts)"
fi

tasks=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(data["tasks"]))
'))

PS3="Select task:"
select task in "${tasks[@]}"
do
    echo "Selected task: $task"
    # Get datasets for selected task
    task_datasets=($(python3 -c "
import yaml
with open('tuning/collections.yaml') as f:
    data = yaml.safe_load(f)
    print(' '.join(data['dataset']['$task']))
"))
    break
done 

PS3="Select dataset:"
select dataset in "${task_datasets[@]}"
do
    echo "Selected dataset: $dataset"
    break
done

sft_ratios=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["sft_ratio"])))
'))

PS3="Select sft ratio:"
select sft_ratio in "${sft_ratios[@]}"
do
    echo "Selected sft ratio: $sft_ratio"
    break
done

base_models=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["base_models"])))
'))

PS3="Select base model:"
select base_model in "${base_models[@]}"
do
    echo "Selected base model: $base_model"
    break
done


pft_types=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["pft_types"])))
'))

PS3="Select pft method:"
select pft_method in "${pft_types[@]}"
do
    echo "Selected pft method: $pft_method"
    break
done

read -p "Do you want to perform training? (y/N): " do_training
read -p "Do you want to perform inference? (y/N): " do_inference
read -p "Do you want to perform evaluation? (y/N): " do_evaluation

# Prepare the command arguments
training_flag=$([ "${do_training,,}" = "y" ] && echo "--do_training" || echo "")
inference_flag=$([ "${do_inference,,}" = "y" ] && echo "--do_inference" || echo "")
evaluation_flag=$([ "${do_evaluation,,}" = "y" ] && echo "--do_evaluation" || echo "")

# Only ask about SFT if training is enabled
do_sft_first="n"
if [ "${do_training,,}" = "y" ]; then
    read -p "Do you want to run SFT first before DPO? (y/N): " do_sft_first
fi    

for train_size in "${train_sizes[@]}"; do
    sft_train_size=$(awk "BEGIN {print $train_size * $sft_ratio}")
    dpo_train_size=$(awk "BEGIN {print $train_size * (1 - $sft_ratio)}")
    
    if [ "$sft_ratio" = "1.0" ]; then
        python tuning/run_sft.py \
            --train_size "$sft_train_size" \
            --model $base_model \
            --dataset $dataset \
            --task $task \
            $training_flag \
            $inference_flag \
            $evaluation_flag
    else

        if [ "${do_sft_first,,}" = "y" ]; then
            echo "Running SFT first (training only)..."
            python tuning/run_sft.py \
                --train_size "$sft_train_size" \
                --model $base_model \
                --dataset $dataset \
                --task $task \
                --do_training
        fi

        python tuning/run_dpo.py \
            --train_size "$dpo_train_size" \
            --model $base_model \
            --dataset $dataset \
            --sft_train_size "$sft_train_size" \
            --task $task \
            --pft_method $pft_method \
            $training_flag \
            $inference_flag \
            $evaluation_flag
    fi
done