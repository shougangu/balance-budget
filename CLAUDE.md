# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for "Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning". Experiments with different ratios of SFT (Supervised Fine-Tuning) and PFT (Preference Fine-Tuning) on LLMs using various data budgets.

Paper: https://arxiv.org/pdf/2502.11284

## Environment Setup

```bash
# Create conda environment
conda create --name <env> --file requirements.txt

# Add IFEval repository to root folder
# Clone from: https://github.com/google-research/google-research/tree/master/instruction_following_eval
```

## Core Commands

### Data Processing
```bash
# Process all datasets (GSM8K and TuluIF for both SFT and preference data)
bash tuning/slurm/data_processing.sh
```

### Training Workflows

**Interactive Mode (recommended for exploration):**
```bash
bash tuning/slurm/run.sh
# Interactive prompts will guide you through:
# 1. Task selection (math/instruction)
# 2. Dataset (gsm8k/tuluif)
# 3. SFT ratio (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
# 4. Base model (llama3-8B, qwen2-7B, etc.)
# 5. PFT method (dpo/kto)
```

**Non-Interactive Mode (for automation/SLURM):**
```bash
# Environment variables or command-line flags
TASK=math DATASET=gsm8k SFT_RATIO=0.5 BASE_MODEL=llama3-8B PFT_METHOD=dpo \
DO_TRAINING=y DO_INFERENCE=y DO_EVALUATION=y \
bash tuning/slurm/run.sh

# Or with flags
bash tuning/slurm/run.sh \
  --task math \
  --dataset gsm8k \
  --sft_ratio 0.5 \
  --base_model llama3-8B \
  --pft_method dpo \
  --do_training \
  --do_inference \
  --do_evaluation
```

**SFT-only training:**
```bash
python tuning/run_sft.py \
  --model llama3-8B \
  --dataset gsm8k \
  --train_size 1000 \
  --task math \
  --do_training --do_inference --do_evaluation
```

**DPO/KTO training (after SFT):**
```bash
python tuning/run_dpo.py \
  --model llama3-8B \
  --dataset gsm8k \
  --train_size 500 \
  --sft_train_size 500 \
  --task math \
  --pft_method dpo \
  --do_training --do_inference --do_evaluation
```

**Sweetspot Pipeline (perplexity-based checkpointing + forked DPO):**
```bash
python tuning/run_sweetspot_pipeline.py \
  --model llama3-8B \
  --dataset tuluif \
  --total_data 10000 \
  --sft_data 5000 \
  --perplexity_thresholds 3.0 2.5 2.0 \
  --task ifeval \
  --do_sft --do_dpo
```

### Evaluation Pipelines

**Pass@K Analysis:**
```bash
# SLURM: tuning/slurm/passk_pipeline.sh
# Notebook: tuning/training/passk_pipeline.ipynb
python tuning/training/passk_pipeline.py
```

**Perplexity Analysis:**
```bash
# SLURM: tuning/slurm/ppl_pipeline.sh
# Notebook: tuning/training/ppl_pipeline.ipynb
python tuning/training/ppl_pipeline.py
```

## Architecture Overview

### Training Pipeline Flow

The codebase implements a modular training pipeline with three main pathways:

1. **SFT-only**: Base model → SFT training → Inference → Evaluation
2. **Sequential SFT→PFT**: Base model → SFT → DPO/KTO → Inference → Evaluation
3. **Sweetspot Pipeline**: Base model → SFT with perplexity checkpoints → Multiple forked DPO branches (one per checkpoint)

### Key Components

**Run Configurations (`tuning/training/config_training.py`):**
- `SFTRunConfig`: Configures SFT runs, generates output paths like `{model}_{dataset_type}-{dataset}-{train_size}`
- `PTRunConfig`: Configures preference training (DPO/KTO), can reference a parent SFTRunConfig for model path resolution
- `DatasetConfig`: Specifies dataset name, type (sft/pt), and train size
- Model outputs saved to: `tuning/models/{run_name}/`

**Training Modules:**
- `tuning/training/sft_training.py`: SFT training with optional perplexity/pass@k callbacks
- `tuning/training/dpo_training.py`: DPO training, loads from SFT checkpoint if `sft_run_config` is provided
- `tuning/training/kto_training.py`: KTO (unpaired preference) training

**Callbacks (Early Stopping):**
- `PerplexityStoppingCallback`: Monitors perplexity on eval set, saves "sweetspot" checkpoints when thresholds are crossed
- `PassAtKStoppingCallback`: Evaluates pass@k during training, stops when target is reached

**Data Loading (`tuning/data/`):**
- `train_dataset.py`: Unified loader, dispatches to task-specific loaders based on `run_config`
- `gsm8k_sft.py`, `gsm8k_pref.py`: GSM8K dataset loaders for SFT and preference data
- `tuluif_sft.py`, `tuluif_pref.py`: TuluIF dataset loaders
- `hf_dataset.py`: Generic HuggingFace dataset wrapper
- Datasets stored in: `tuning/data/datasets/`

**Inference (`tuning/inference/`):**
- `ifeval_inference.py`: IFEval task inference (instruction following)
- `k_beams.py`, `k_beams_hf.py`: Beam search utilities for pass@k evaluation
- Uses vLLM for efficient batch inference

**Evaluation (`tuning/evaluation/`):**
- `ifeval_evaluate.py`: Evaluates IFEval responses using Google's evaluation library
- `gsm8k_evaluate.py`: Evaluates GSM8K math reasoning
- Outputs saved to: `tuning/outputs/{task}/`

### Model Configuration

**Supported base models** (defined in `tuning/config.py`):
- `llama3-8B`: Meta-Llama-3.1-8B (unsloth)
- `qwen2-7B`, `qwen2-3B`, `qwen2-2B`, `qwen2-14B`: Qwen2.5 variants
- `llama3-3B`, `llama3-1B`: Llama-3.2 small variants

**LoRA Configuration:**
- Default r=32, alpha=32, targets all attention and MLP projections
- Qwen models require additional targets: `embed_tokens`, `lm_head`

**Training Hyperparameters:**
- Effective batch size: 16 (via gradient accumulation)
- SFT: 2 epochs, lr=5e-5, AdamW 8-bit
- DPO: 2 epochs, lr=5e-6, beta=1.0
- Uses unsloth for memory-efficient training

### Sweetspot Pipeline Architecture

The sweetspot pipeline (`run_sweetspot_pipeline.py`) implements a forking strategy:

1. **SFT Phase**: Train with perplexity callback, save checkpoints at each threshold crossing
   - Checkpoint metadata stored in: `checkpoint-sweetspot-{threshold}/sweetspot_metadata.json`
   - Metadata includes: threshold value, global step, checkpoint path

2. **DPO Forking**: For each sweetspot checkpoint, launch independent DPO training
   - DPO uses remaining data budget: `total_data - sft_data_used`
   - Each fork saved to: `{model}_sweetspot-ppl-{threshold}_dpo-{dataset}-{dpo_size}/`

3. **Parallel Evaluation**: Run inference/evaluation on all DPO outputs

### Configuration Files

- `tuning/collections.yaml`: Defines available tasks, datasets, SFT ratios, models, and PFT methods
- `tuning/config.py`: Global paths and HuggingFace model mappings
- `requirements.txt`: Python dependencies (conda-compatible)

### SLURM Integration

SLURM scripts in `tuning/slurm/`:
- Use `#SBATCH` directives for resource allocation
- Set environment variables before calling run scripts
- Typical resources: 1 GPU, 2-4 hours for small-scale experiments

### Important Notes

- **IFEval Dependency**: The instruction_following_eval repository must be cloned into the root directory for IFEval tasks to work
- **vLLM Memory**: Large models may encounter vLLM memory issues; adjust `max_seq_length` or use smaller batch sizes
- **W&B Integration**: All training runs log to wandb project "tuning"
- **Model Paths**: Models are loaded via SFTRunConfig/PTRunConfig which automatically construct paths; don't hardcode model directories
- **Train Size in run.sh**: Modify the `train_sizes` array in `tuning/slurm/run.sh` to experiment with different data budgets
