"""
Sweetspot Pipeline: SFT with checkpoint forking -> DPO on each checkpoint.

This pipeline implements the "Fork Strategy":
1. Run SFT with perplexity thresholds, saving checkpoints at each sweetspot
2. After SFT completes, run DPO on each saved checkpoint
3. DPO uses (total_data - sft_data_used) for preference training

Usage:
    python -m tuning.run_sweetspot_pipeline \
        --model llama3-8B \
        --dataset tuluif \
        --total_data 10000 \
        --sft_data 5000 \
        --perplexity_thresholds 3.0 2.5 2.0 \
        --task ifeval
"""

import wandb
import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Optional

from tuning.training.config_training import (
    ModelLoadConfig, LoraConfig, PTRunConfig, DPOTrainingConfig, 
    DatasetConfig, SFTRunConfig, TrainingArgumentsConfig
)
from tuning.config import HF_MODEL_MAP, MODELS_DIR
from tuning.training.sft_training import train_model_sft
from tuning.training.dpo_training import train_model_dpo
from tuning.run_inference import run_inference
from tuning.run_evaluation import run_evaluation

import torch
import gc
import time


def find_sweetspot_checkpoints(output_dir: str) -> List[dict]:
    """Find all sweetspot checkpoints in the output directory."""
    checkpoints = []
    pattern = os.path.join(output_dir, "checkpoint-sweetspot-*", "sweetspot_metadata.json")
    
    for metadata_path in glob.glob(pattern):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        checkpoints.append(metadata)
    
    # Sort by threshold value (descending - higher perplexity first)
    checkpoints.sort(key=lambda x: x["threshold_value"], reverse=True)
    return checkpoints


def run_sft_with_sweetspots(
    model: str,
    dataset: str,
    sft_data_size: int,
    perplexity_thresholds: List[float],
    task: str,
    lora_config: LoraConfig,
    model_load_config: ModelLoadConfig,
    training_args: TrainingArgumentsConfig,
) -> str:
    """
    Run SFT training with perplexity sweetspot checkpointing.
    
    Returns:
        output_dir: Path to the SFT output directory containing sweetspot checkpoints
    """
    dataset_config = DatasetConfig(
        dataset=dataset,
        dataset_type="sft",
        train_size=sft_data_size,
    )
    
    run_config = SFTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[model],
        model_name=model,
        task_name=task,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    
    print(f"\n{'='*60}")
    print(f"[Pipeline] Starting SFT with sweetspot checkpointing")
    print(f"[Pipeline] Model: {model}")
    print(f"[Pipeline] Dataset: {dataset} (SFT size: {sft_data_size})")
    print(f"[Pipeline] Perplexity thresholds: {perplexity_thresholds}")
    print(f"[Pipeline] Output: {run_config.output_dir}")
    print(f"{'='*60}\n")
    
    run = wandb.init(
        name=f"{run_config.run_name}_sweetspot", 
        project="tuning", 
        reinit=True
    )
    
    with run:
        model_obj, tokenizer, trainer = train_model_sft(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            perplexity_thresholds=perplexity_thresholds,
        )
        
        # Cleanup
        del model_obj
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    return run_config.output_dir


def run_dpo_on_checkpoint(
    checkpoint_metadata: dict,
    dataset: str,
    dpo_data_size: int,
    task: str,
    model_name: str,
    lora_config: LoraConfig,
    model_load_config: ModelLoadConfig,
    training_args: DPOTrainingConfig,
) -> str:
    """
    Run DPO training on a specific sweetspot checkpoint.
    
    Args:
        checkpoint_metadata: Metadata dict from sweetspot checkpoint
        dpo_data_size: Amount of data to use for DPO (total - sft_used)
        
    Returns:
        output_dir: Path to the DPO output directory
    """
    checkpoint_path = checkpoint_metadata["checkpoint_path"]
    threshold = checkpoint_metadata["threshold_value"]
    step = checkpoint_metadata["global_step"]
    
    # Create DPO dataset config
    dataset_config = DatasetConfig(
        dataset=dataset,
        dataset_type="pt",
        train_size=dpo_data_size,
    )
    
    # Create a pseudo SFT run config that points to the sweetspot checkpoint
    # This is used to construct the model path for DPO
    sft_run_config = SFTRunConfig(
        model_name=model_name,
        model_name_hf=checkpoint_path,  # Point to the actual checkpoint
        dataset_config=DatasetConfig(
            dataset=dataset,
            dataset_type="sft",
            train_size=step,  # Use step as proxy for data seen
        ),
    )
    # Override the output_dir property by using checkpoint_path directly
    
    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=checkpoint_path,  # Load from checkpoint
        model_name=model_name,
        sft_run_config=None,  # We'll handle model path manually
        task_name=task,
        pft_method="dpo",
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    
    # Custom run name and output dir for this sweetspot
    custom_run_name = f"{model_name}_sweetspot-ppl-{threshold:.2f}_dpo-{dataset}-{dpo_data_size}"
    custom_output_dir = f"{MODELS_DIR}/{custom_run_name}"
    
    print(f"\n{'='*60}")
    print(f"[Pipeline] Starting DPO on sweetspot checkpoint")
    print(f"[Pipeline] Checkpoint: {checkpoint_path}")
    print(f"[Pipeline] Threshold: perplexity={threshold}, step={step}")
    print(f"[Pipeline] DPO data size: {dpo_data_size}")
    print(f"[Pipeline] Output: {custom_output_dir}")
    print(f"{'='*60}\n")
    
    run = wandb.init(
        name=custom_run_name,
        project="tuning",
        reinit=True,
        config={
            "sweetspot_threshold": threshold,
            "sweetspot_step": step,
            "checkpoint_path": checkpoint_path,
            "dpo_data_size": dpo_data_size,
        }
    )
    
    with run:
        train_model_dpo_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=custom_output_dir,
            dataset=dataset,
            dpo_data_size=dpo_data_size,
            model_name=model_name,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
        )
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return custom_output_dir


def train_model_dpo_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    dataset: str,
    dpo_data_size: int,
    model_name: str,
    lora_config: LoraConfig,
    model_load_config: ModelLoadConfig,
    training_args: DPOTrainingConfig,
):
    """
    Train DPO model from a specific checkpoint path.
    
    This is a specialized version that loads from an arbitrary checkpoint path
    rather than constructing it from run configs.
    """
    from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
    from trl import DPOTrainer, DPOConfig
    from tuning.data.train_dataset import get_train_dataset
    from tuning.utils.utils import apply_chat_template_pt, chat_template_func
    from tuning.training.config_training import dpo_batch_size, effective_batch_size
    
    PatchDPOTrainer()
    
    train_batch_size = dpo_batch_size(dpo_data_size)
    gradient_accumulation_steps = effective_batch_size(dpo_data_size) // train_batch_size
    
    # Load DPO dataset using a minimal run config
    dataset_config = DatasetConfig(
        dataset=dataset,
        dataset_type="pt",
        train_size=dpo_data_size,
    )
    
    # Create a minimal PTRunConfig just to load the dataset
    minimal_run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=checkpoint_path,
        model_name=model_name,
        run_type="pt",
    )
    dpo_dataset = get_train_dataset(minimal_run_config)
    
    print(f"[DPO] Loading model from checkpoint: {checkpoint_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=model_load_config.max_seq_length,
        dtype=model_load_config.dtype,
        load_in_4bit=model_load_config.load_in_4bit,
    )
    
    dpo_dataset = apply_chat_template_pt(tokenizer, dpo_dataset)
    
    print(f"[DPO] Dataset: {dpo_dataset}")
    print(f"[DPO] Sample: {dpo_dataset['train'][0]}")
    
    if model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
    )
    
    print(f"[DPO] Model loaded - {type(model)}")
    print(f"[DPO] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=chat_template_func(tokenizer),
        beta=training_args.beta,
        train_dataset=dpo_dataset["train"],
        eval_dataset=dpo_dataset["test"],
        max_length=model_load_config.max_seq_length,
        args=DPOConfig(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            warmup_ratio=training_args.warmup_ratio,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            do_eval=training_args.do_eval,
            eval_strategy=training_args.eval_strategy,
            eval_steps=training_args.eval_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            report_to=training_args.report_to,
            logging_steps=training_args.logging_steps,
            output_dir=output_dir,
            save_strategy="no",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=42,
        )
    )
    
    trainer_stats = trainer.train()
    
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    
    args = trainer.args.to_dict()
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)
    
    # Save pipeline metadata
    pipeline_metadata = {
        "source_checkpoint": checkpoint_path,
        "dpo_data_size": dpo_data_size,
        "dataset": dataset,
    }
    with open(f"{output_dir}/pipeline_metadata.json", "w") as f:
        json.dump(pipeline_metadata, f, indent=4)


def run_pipeline(
    model: str,
    dataset: str,
    total_data: int,
    sft_data: int,
    perplexity_thresholds: List[float],
    task: str,
    do_sft: bool = True,
    do_dpo: bool = True,
    do_inference: bool = False,
    do_evaluation: bool = False,
    sft_output_dir: Optional[str] = None,
):
    """
    Run the complete sweetspot pipeline.
    
    Args:
        model: Model name (e.g., "llama3-8B")
        dataset: Dataset name (e.g., "tuluif")
        total_data: Total data budget
        sft_data: Data to use for SFT phase
        perplexity_thresholds: List of perplexity thresholds for checkpointing
        task: Task name (e.g., "ifeval")
        do_sft: Whether to run SFT phase
        do_dpo: Whether to run DPO phase on all checkpoints
        sft_output_dir: If skipping SFT, provide the output dir to find checkpoints
    """
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    sft_training_args = TrainingArgumentsConfig()
    dpo_training_args = DPOTrainingConfig()
    
    # Calculate DPO data size (remaining budget after SFT)
    dpo_data_size = total_data - sft_data
    
    print(f"\n{'#'*60}")
    print(f"# SWEETSPOT PIPELINE")
    print(f"# Model: {model}")
    print(f"# Dataset: {dataset}")
    print(f"# Total data budget: {total_data}")
    print(f"# SFT data: {sft_data}")
    print(f"# DPO data (per branch): {dpo_data_size}")
    print(f"# Perplexity thresholds: {perplexity_thresholds}")
    print(f"{'#'*60}\n")
    
    # Phase 1: SFT with sweetspot checkpointing
    if do_sft:
        sft_output_dir = run_sft_with_sweetspots(
            model=model,
            dataset=dataset,
            sft_data_size=sft_data,
            perplexity_thresholds=perplexity_thresholds,
            task=task,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=sft_training_args,
        )
        print(f"\n[Pipeline] SFT completed. Output: {sft_output_dir}")
        time.sleep(10)  # Allow GPU memory to settle
    
    if sft_output_dir is None:
        raise ValueError("sft_output_dir must be provided if do_sft=False")
    
    # Find all sweetspot checkpoints
    checkpoints = find_sweetspot_checkpoints(sft_output_dir)
    print(f"\n[Pipeline] Found {len(checkpoints)} sweetspot checkpoints:")
    for ckpt in checkpoints:
        print(f"  - perplexity={ckpt['threshold_value']}, step={ckpt['global_step']}, path={ckpt['checkpoint_path']}")
    
    # Phase 2: Run DPO on each checkpoint
    if do_dpo and dpo_data_size > 0:
        dpo_outputs = []
        for i, checkpoint in enumerate(checkpoints):
            print(f"\n[Pipeline] DPO Branch {i+1}/{len(checkpoints)}")
            
            output_dir = run_dpo_on_checkpoint(
                checkpoint_metadata=checkpoint,
                dataset=dataset,
                dpo_data_size=dpo_data_size,
                task=task,
                model_name=model,
                lora_config=lora_config,
                model_load_config=model_load_config,
                training_args=dpo_training_args,
            )
            dpo_outputs.append({
                "sweetspot": checkpoint["threshold_value"],
                "output_dir": output_dir,
            })
            
            time.sleep(10)  # Allow GPU memory to settle
        
        print(f"\n[Pipeline] DPO completed on all checkpoints:")
        for out in dpo_outputs:
            print(f"  - sweetspot={out['sweetspot']}: {out['output_dir']}")
    
    # Phase 3: Inference and Evaluation (optional)
    if do_inference or do_evaluation:
        print("\n[Pipeline] Running inference/evaluation on DPO outputs...")
        # TODO: Implement batch inference and evaluation
        # This would iterate over dpo_outputs and run inference/eval on each
    
    print(f"\n{'#'*60}")
    print("# PIPELINE COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweetspot Pipeline: SFT -> DPO")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., llama3-8B)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., tuluif)")
    parser.add_argument("--total_data", type=int, required=True,
                        help="Total data budget")
    parser.add_argument("--sft_data", type=int, required=True,
                        help="Data to use for SFT phase")
    parser.add_argument("--perplexity_thresholds", type=float, nargs="+", required=True,
                        help="Perplexity thresholds for checkpointing (e.g., 3.0 2.5 2.0)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., ifeval)")
    
    parser.add_argument("--do_sft", action="store_true",
                        help="Run SFT phase")
    parser.add_argument("--do_dpo", action="store_true",
                        help="Run DPO phase on all checkpoints")
    parser.add_argument("--do_inference", action="store_true",
                        help="Run inference on outputs")
    parser.add_argument("--do_evaluation", action="store_true",
                        help="Run evaluation on outputs")
    
    parser.add_argument("--sft_output_dir", type=str, default=None,
                        help="Path to existing SFT output (skip SFT phase)")
    
    args = parser.parse_args()
    
    run_pipeline(
        model=args.model,
        dataset=args.dataset,
        total_data=args.total_data,
        sft_data=args.sft_data,
        perplexity_thresholds=args.perplexity_thresholds,
        task=args.task,
        do_sft=args.do_sft,
        do_dpo=args.do_dpo,
        do_inference=args.do_inference,
        do_evaluation=args.do_evaluation,
        sft_output_dir=args.sft_output_dir,
    )
