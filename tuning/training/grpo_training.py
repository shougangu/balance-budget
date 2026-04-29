# ABOUTME: GRPO (RLVR) training using TRL's GRPOTrainer with standard HF/PEFT.
# ABOUTME: Mirrors dpo_training.py pattern with verifiable reward functions.

import torch
import wandb

from tuning.config import MODELS_DIR
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import PTRunConfig, LoraConfig, ModelLoadConfig, GRPOTrainingConfig
from tuning.training.callback_utils import CompletionsIntervalCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.model_utils import load_model_with_lora, save_trained_model, top_layer_indices

from tuning.utils.utils import chat_template_func
from trl import GRPOTrainer, GRPOConfig
from typing import Callable, List
from tuning.config import HF_MODEL_MAP
import subprocess


def train_model_grpo(
    run_config: PTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: GRPOTrainingConfig = None,
    reward_funcs: List[Callable] = None,
    passk_config = None,
    primary_eval = None,
    monitor_evals = None,
    initial_global_step = None,
    lora_layers_fraction = 1.0,
):
    # Resolve model path: SFT checkpoint or base HF model
    if run_config.sft_run_config:
        if run_config.sft_run_config.dataset_config.dynamic_path:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.dataset_config.dynamic_path}"
        else:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.run_name}"
    else:
        model_path = run_config.model_name_hf

    raw_dataset = get_train_dataset(run_config)

    layers = None
    if lora_layers_fraction < 1.0:
        layers = top_layer_indices(run_config.model_name_hf, lora_layers_fraction)

    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_with_lora(
        model_path=model_path,
        model_name=run_config.model_name,
        model_load_config=model_load_config,
        lora_config=lora_config,
        use_unsloth=False,
        layers_to_transform=layers,
    )
    tokenizer = chat_template_func(tokenizer)

    callbacks = []

    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=model_path,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals or [],
        )
        callbacks.append(passk_callback)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["test"],
        callbacks=callbacks if callbacks else None,
        args=GRPOConfig(
            **training_args.to_hf_args(output_dir=run_config.output_dir),
        ),
    )

    if trainer.log_completions:
        trainer.add_callback(CompletionsIntervalCallback(trainer, interval=32))

    for cb in callbacks or []:
        if isinstance(cb, PassAtKStoppingCallback):
            if hasattr(trainer, 'vllm_generation'):
                cb.set_trainer_vllm(trainer.vllm_generation.llm)
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's vLLM engine")
            cb._accelerator = trainer.accelerator

    # Swap the default WandbCallback for one that bridges train/global_step across runs.
    if initial_global_step:
        from transformers.integrations import WandbCallback
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        trainer.pop_callback(WandbCallback)
        trainer.add_callback(OffsetAwareWandbCallback(initial_global_step))

    try:
        trainer_stats = trainer.train()
    except KeyboardInterrupt:
        if wandb.run:
            wandb.run.tags = list(wandb.run.tags) + ["interrupted"]
        raise
    except torch.cuda.OutOfMemoryError:
        print(subprocess.check_output("nvidia-smi").decode())
        if wandb.run:
            wandb.run.tags = list(wandb.run.tags) + ["oom"]
        raise

    save_trained_model(model, tokenizer, trainer, run_config.output_dir)

    return model, tokenizer, trainer, callbacks
