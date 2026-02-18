
import torch
import wandb
from unsloth import PatchDPOTrainer
from tuning.config import MODELS_DIR
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import PTRunConfig, LoraConfig, ModelLoadConfig, DatasetConfig, DPOTrainingConfig, SFTRunConfig, PassAtKConfig, PerplexityConfig, dpo_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.model_utils import load_model_with_lora, save_trained_model
from tuning.utils.utils import chat_template_func
from trl import DPOTrainer, DPOConfig
from typing import List, Optional
from tuning.config import HF_MODEL_MAP, resolve_chat_template
import subprocess
PatchDPOTrainer()


def train_model_dpo(
    run_config: PTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: DPOTrainingConfig = None,
    perplexity_config = None,  # PerplexityConfig object
    perplexity_test_dataset = None,  # SFT-formatted test dataset for perplexity eval
    passk_config = None,  # PassAtKConfig object
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

    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_with_lora(
        model_path=model_path,
        model_name=run_config.model_name,
        model_load_config=model_load_config,
        lora_config=lora_config,
    )
    chat_template = resolve_chat_template(run_config.model_name, run_config.chat_template)
    tokenizer = chat_template_func(tokenizer, chat_template=chat_template)

    callbacks = []
    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=model_path,
        )
        callbacks.append(passk_callback)

    if perplexity_config is not None and perplexity_config.enabled:
        perplexity_callback = PerplexityStoppingCallback(
            config=perplexity_config,
            test_dataset=perplexity_test_dataset,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
        )
        callbacks.append(perplexity_callback)

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        tokenizer = tokenizer,
        train_dataset = raw_dataset["train"],
        eval_dataset = raw_dataset["test"],
        max_length = model_load_config.max_seq_length,
        callbacks = callbacks if callbacks else None,
        args = DPOConfig(**training_args.to_hf_args(output_dir=run_config.output_dir)),
    )

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

if __name__ == "__main__":

    model = "llama3-8B"
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()

    dataset_config = DatasetConfig(
        dataset="tuluif",
        dataset_type="pt",
        train_size=5000,
    )

    sft_run_config = SFTRunConfig(
        model_name=model,
        model_name_hf=HF_MODEL_MAP[model],
        dataset_config=DatasetConfig(
            dataset="tuluif",
            dataset_type="sft",
            train_size=5000,
        ),
    )

    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[model],
        model_name=model,
        sft_run_config=sft_run_config,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )

    run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

    with run:
        # Configure pass@k evaluation
        passk_config = PassAtKConfig(
            target_pass_at_k=[1.2],
            k_values=[1],
            n_samples=1,
            num_prompts=32,
            temperature=0.7,
            strict=True,
            enabled=True,
        )

        train_model_dpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            # perplexity_thresholds=[0.1],  # Set to a value like 1.5 to enable
            # passk_config=passk_config,
        )
