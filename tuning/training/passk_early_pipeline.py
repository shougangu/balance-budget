import os
# Save full GPU list for multi-GPU inference workers, then restrict training to GPU 0.
# Must happen before any CUDA/torch imports.
_ALL_VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _ALL_VISIBLE_GPUS:
    # Stash the full list so PassAtKCallback workers can find all GPUs
    os.environ["CUDA_VISIBLE_DEVICES_ALL"] = _ALL_VISIBLE_GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = _ALL_VISIBLE_GPUS.split(",")[0]

from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, PTRunConfig, DPOTrainingConfig, TrainingArgumentsConfig, PassAtKConfig, sft_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.utils.utils import chat_template_func
import json
import sys
from datasets import load_from_disk
from typing import List, Optional, Union
from pathlib import Path
from tuning.config import DATASETS_DIR, HF_MODEL_MAP
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
from tuning.utils.gpu import cleanup_gpu
from tuning.training.wandb_utils import get_early_pairs, early_pair_tag

MODEL_TO_GPU_1 = {
    "llama3-1B": 0.75,
    "llama3-3B": 0.62,
    "llama3-8B": 0.68,
    "qwen2-3B": 0.62
}
MODEL_TO_GPU_2 = {
    "llama3-1B": 0.7,
    "llama3-3B": 0.64, # this is really tight, can reach 80.7/81.6
    "llama3-8B": 0.45, 
    "qwen2-3B": 0.6
}

if __name__ == '__main__':
    MODEL = "llama3-3B"
    gpu_utilisation_1 = MODEL_TO_GPU_1[MODEL]
    gpu_utilisation_2 = MODEL_TO_GPU_2[MODEL]
    total_train_size = 10240  # 29980

    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "sft",
        train_size = total_train_size, # 29980
    )

    run_config = SFTRunConfig(
        chat_template="chatml",
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[MODEL],  # Use HuggingFace model name, not local path
        model_name = MODEL,  # Base model name for output directory construction
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    print(f"Run config: {run_config}")

    lora_config = LoraConfig()

    model_load_config = ModelLoadConfig()
    # model_load_config.max_seq_length = 4096

    training_args = TrainingArgumentsConfig()

    # ---------------------------------------------
    training_args.eval_steps = 64
    training_args.per_device_train_batch_size = 16
    training_args.gradient_accumulation_steps = 1
    # ---------------------------------------------

    passk_config = PassAtKConfig( # this is just to dynamically view the pass@1 of ifeval
        target_pass_at_k=[0.1, 0.15, 0.2,0.25,0.3, 0.9],
         # ---------------------------------------------
        early_tuples = [(1, 0.02), (2,0.02), (3,0.02), (4,0.02), (5,0.02)], #####
        k_values=[32,16,8,4,2,1], #####
        n_samples=32, #/####
        num_prompts=270, #####
        vllm_gpu_memory_utilization=gpu_utilisation_1,
        # ---------------------------------------------
        temperature=0.7,
        strict=True,
        enabled=True,
        num_inference_gpus=4,
    )

    sft_early_pairs = get_early_pairs(passk_config)
    run = wandb.init(
        name=run_config.model_name,
        project="tuning", 
        job_type="sft",
        tags=["passk", "sft", early_pair_tag(sft_early_pairs)],
        # Optional: Pass config here so it's logged even if training crashes early
        config={
            "pipeline_type": "passk",
            "stage": "sft",
            "early_pairs": sft_early_pairs,
        },
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
    if not Path(metadata_file).exists():
        print(f"Metadata file {metadata_file} does not exist. Exiting.")
        sys.exit(1)
    checkpoints = [] 
    with open(metadata_file, "r") as f:
        for line in f:
            checkpoints.append(json.loads(line))
    print(checkpoints)

    del model, tokenizer, trainer, callbacks
    cleanup_gpu()
    print(subprocess.check_output("nvidia-smi").decode())

    for checkpoint in checkpoints:    
        model_name = Path(checkpoint["checkpoint_path"]).name
        data = total_train_size - checkpoint["data_points_seen"] 
        model_load_config = ModelLoadConfig()
        training_args = DPOTrainingConfig()

        # ======================================= 
        training_args.per_device_train_batch_size = 4  
        training_args.gradient_accumulation_steps = 4  
        training_args.eval_steps = 64
        # ======================================= 
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
            chat_template="chatml",
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
            # ----------------------------------------
            k_values=[1], #####
            n_samples=1, #####
            num_prompts=541,
            vllm_gpu_memory_utilization=gpu_utilisation_2, #0.58
            # ----------------------------------------
            temperature=0.5,
            strict=True,
            enabled=True,
        )
        # lora_config.use_gradient_checkpointing = True  # Reduce activation memory
        
        dpo_early_pairs = get_early_pairs(passk_config)
        run = wandb.init(
            name=run_config.model_name,
            project="tuning",
            job_type="dpo",
            tags=["passk", "dpo", early_pair_tag(dpo_early_pairs)],
            config={
                "pipeline_type": "passk",
                "stage": "dpo",
                "early_pairs": dpo_early_pairs,
            },
            reinit=True,
        )
        with run:
            model, tokenizer, trainer, _ = train_model_dpo(
                run_config = run_config,
                lora_config = lora_config,
                model_load_config = model_load_config,
                training_args = training_args,
                passk_config = passk_config,
                # perplexity_thresholds= [0.1] # dummy value to periodically check perplexities too
            )
        
        try:
            wandb.finish()
        except Exception:
            pass
        
        del model, tokenizer, trainer
        cleanup_gpu()
        print(subprocess.check_output("nvidia-smi").decode())
