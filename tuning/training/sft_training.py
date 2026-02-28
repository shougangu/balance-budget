
from trl import SFTTrainer
from transformers import TrainingArguments
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, PassAtKConfig, PerplexityConfig, DatasetConfig, IFEvalConfig, sft_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.eval_strategy import IFEvalStrategy
from tuning.training.model_utils import load_model_with_lora, save_trained_model
from tuning.utils.utils import chat_template_func, apply_chat_template, get_response_delimiters
from typing import List, Optional
from pathlib import Path
from tuning.config import HF_MODEL_MAP, MODELS_DIR
import torch
import wandb
import subprocess

def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
    perplexity_config = None,  # PerplexityConfig object
    passk_config = None,  # PassAtKConfig object
    ifeval_config = None,  # IFEvalConfig object
):
    dataset = get_train_dataset(run_config)
    raw_eval_dataset = dataset["test"]

    model, tokenizer = load_model_with_lora(
        model_path=run_config.model_name_hf,
        model_name=run_config.model_name,
        model_load_config=model_load_config,
        lora_config=lora_config,
    )
    tokenizer = chat_template_func(tokenizer)

    dataset = apply_chat_template(tokenizer, dataset)
    print(f"Example SFT input:\n{dataset['train'][0]['text']}")

    callbacks = []
    if passk_config is not None and passk_config.enabled:
        eval_cfg = ifeval_config or IFEvalConfig()
        ifeval_strategy = IFEvalStrategy(
            k_values=eval_cfg.k_values,
            n_samples=eval_cfg.n_samples,
            strict=eval_cfg.strict,
            num_prompts=eval_cfg.num_prompts,
            tokenizer=tokenizer,
        )
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=run_config.model_name_hf,
            primary_eval=ifeval_strategy,
        )
        callbacks.append(passk_callback)

    if perplexity_config is not None and perplexity_config.enabled:
        perplexity_callback = PerplexityStoppingCallback(
            config=perplexity_config,
            test_dataset=raw_eval_dataset,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
        )
        callbacks.append(perplexity_callback)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = model_load_config.max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        callbacks = callbacks if callbacks else None,
        args = TrainingArguments(
            **training_args.to_hf_args(output_dir=run_config.output_dir),
        ),
    )

    # Mask non-response tokens in labels using template-specific delimiters
    from unsloth import train_on_responses_only
    train_on_responses_only(trainer, **get_response_delimiters())

    print(trainer.args.to_dict())

    # Resume from checkpoint if it exists
    resume_from_checkpoint = None
    if Path(run_config.output_dir).exists():
        checkpoints = list(Path(run_config.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
            print(f"[SFT] Resuming from checkpoint: {resume_from_checkpoint}")

    try:
        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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

    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "sft",
        train_size = 10000, # 29980
    )

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[model],  # Use HuggingFace model name, not local path
        model_name = model,  # Base model name for output directory construction
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )

    print(run_config)

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = 4096
    training_args = TrainingArgumentsConfig()

    # Enable checkpointing for main runs
    # training_args.save_strategy = "steps"
    # training_args.save_steps = 2
    # training_args.save_total_limit = 3
    # training_args.load_best_model_at_end = False
    # training_args.dataloader_drop_last = False


    model, tokenizer, trainer = train_model_sft(
        run_config = run_config,
        lora_config = lora_config,
        model_load_config = model_load_config,
        training_args = training_args,
        # perplexity_thresholds = perplexity_thresholds,
    )
