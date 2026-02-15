from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, PTRunConfig, DPOTrainingConfig, TrainingArgumentsConfig, PassAtKConfig, sft_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.utils.utils import apply_chat_template, chat_template_func
import json
import sys
from datasets import load_from_disk
from typing import List, Optional, Union
from pathlib import Path
from tuning.config import DATASETS_DIR, HF_MODEL_MAP
import os
from tuning.training.config_training import DatasetConfig, SFTRunConfig
from tuning.config import MODELS_DIR
from tuning.training.sft_training import train_model_sft
from tuning.training.dpo_training import train_model_dpo
import subprocess
import importlib
import torch
import json
import wandb
import gc


if __name__ == '__main__':
    MODEL = "llama3-1B"
    total_train_size = 10000  # 29980

    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "sft",
        train_size = total_train_size, # 29980
    )

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[MODEL],  # Use HuggingFace model name, not local path
        model_name = MODEL,  # Base model name for output directory construction
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )

    lora_config = LoraConfig()
    # lora_config.use_gradient_checkpointing = True  # Reduce activation memory

    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = 4096

    training_args = TrainingArgumentsConfig()
    training_args.eval_steps = 64
    training_args.per_device_train_batch_size = 16
    training_args.gradient_accumulation_steps = 1

    passk_config = PassAtKConfig( # this is just to dynamically view the pass@1 of ifeval
        target_pass_at_k=[0.1, 0.15, 0.2,0.25,0.3, 0.9],
        k_values=[4],
        n_samples=8,
        num_prompts=541,
        temperature=0.7,
        strict=True,
        enabled=True,
        vllm_gpu_memory_utilization=0.75
    )

    

    run = wandb.init(
        name=run_config.run_name, 
        project="tuning", 
        # reinit=True,
        # Optional: Pass config here so it's logged even if training crashes early
        config=run_config.__dict__ if hasattr(run_config, "__dict__") else {} 
    )

    with run:
        model, tokenizer, trainer, callbacks = train_model_sft(
            run_config = run_config,
            lora_config = lora_config,
            model_load_config = model_load_config,
            training_args = training_args,
            passk_config = passk_config
        )   

    passk_callback = next(c for c in callbacks if isinstance(c, PassAtKStoppingCallback))
    metadata_file = passk_callback.metadata_path
    checkpoints = []
    with open(metadata_file, "r") as f:
        for line in f:
            checkpoints.append(json.loads(line))
    print(checkpoints)

    del model, tokenizer, trainer, callbacks # this deletes the references to such objects
    gc.collect() # then we force the GC
    torch.cuda.empty_cache() # and we release the GPU CUDA cache
    print(subprocess.check_output("nvidia-smi").decode()) # check GPU memory after cleanup

    for checkpoint in checkpoints:    
        model_name = Path(checkpoint["checkpoint_path"]).name
        data = total_train_size - checkpoint["data_points_seen"] 
        model_load_config = ModelLoadConfig()
        training_args = DPOTrainingConfig()
        # training_args.eval_strategy = "epoch"  # Evaluates at end of each epoch
        # training_args.load_best_model_at_end = False  # Keep latest model
        # training_args.report_to = []  # Disable all logging integrations including W&B

        # Memory-optimized for H100: DPO needs 2 models (train + ref) simultaneously
        # With batch_size=4, total memory: ~40GB (models) + ~15GB (activations) = ~55GB ✓
        training_args.per_device_train_batch_size = 4  # Reduced from 16
        training_args.gradient_accumulation_steps = 4  # Maintain effective batch of 16
        training_args.eval_steps = 8
        dataset_config = DatasetConfig(
            dataset = "tuluif",
            dataset_type = "pt",
            train_size = data,
        )
        sft_run_config = SFTRunConfig(
            dataset_config = DatasetConfig(
                dataset = "tuluif",
                dataset_type = "sft",
                train_size = checkpoint["data_points_seen"],
                dynamic_path = model_name
            ),
            model_name = MODEL,
            model_name_hf = HF_MODEL_MAP[MODEL], 
            task_name = "ifeval"
        )
        run_config = PTRunConfig(
            dataset_config = dataset_config,
            model_name_hf = HF_MODEL_MAP[MODEL],  
            model_name = MODEL,  
            sft_run_config = sft_run_config,
            task_name = "ifeval",
            pft_method = "dpo",
            do_training = True
        )
        passk_config = PassAtKConfig( # this is just to dynamically view the pass@1 of ifeval
            target_pass_at_k=[1.2],
            k_values=[1],
            n_samples=1,
            num_prompts=541,
            temperature=0.5,
            strict=True,
            enabled=True,
        )
        
        model, tokenizer, trainer, _ = train_model_dpo(
            run_config = run_config,
            lora_config = lora_config,
            model_load_config = model_load_config,
            training_args = training_args,
            passk_config = passk_config,
            vllm_gpu_memory_utilization=0.6,
            # perplexity_thresholds= [0.1] # dummy value to periodically check perplexities too
        )
        
        # Clean up W&B after training
        try:
            wandb.finish()
        except Exception:
            pass
        
        del model, tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()
        print(subprocess.check_output("nvidia-smi").decode()) # check GPU memory after cleanup

