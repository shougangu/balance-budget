
from trl import SFTTrainer, SFTConfig
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, PassAtKConfig, PerplexityConfig, DatasetConfig, sft_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.model_utils import (
    disable_loss_kwargs_if_unsupported,
    load_model_with_lora,
    save_trained_model,
)
from tuning.training.callback_utils import (
    OffsetAwareWandbCallback,
    remove_default_wandb_callback,
)
from tuning.utils.utils import (
    chat_template_func,
    apply_chat_template,
    tokenize_sft_dataset,
)
from typing import List, Optional
from pathlib import Path
from tuning.config import HF_MODEL_MAP, MODELS_DIR
import torch
import wandb
import subprocess


def padding_free_status(model, args, full_finetune):
    """Describe whether padding-free batching survived, and why it did not.

    Padding-free concatenates a micro-batch instead of padding every sequence to the
    batch maximum, so losing it roughly doubles the cost of a step. Only Unsloth's
    text-model path provides it, and it switches off silently, so name the reason.
    """
    if getattr(args, "padding_free", False):
        return "on"
    if full_finetune:
        return "off (full fine-tune runs the plain HF path, which has no flash-attention build)"
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) or []
    is_vlm = any(name.endswith("ForConditionalGeneration") for name in architectures)
    if is_vlm or hasattr(config, "vision_config"):
        return "off (vision-language architecture; Unsloth refuses padding-free)"
    return "off (unexpected: check for a data_collator or processor passed to SFTTrainer)"


def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
    perplexity_config = None,  # PerplexityConfig object
    passk_config = None,  # PassAtKConfig object
    primary_eval = None,  # Pre-built EvalStrategy
    monitor_evals = None,  # Additional EvalStrategy list
    pipeline_args = None,  # Parsed pipeline CLI args, for callback-triggered live dispatch
):
    dataset = get_train_dataset(run_config)
    raw_eval_dataset = dataset["test"]

    model, tokenizer = load_model_with_lora(
        model_path=run_config.model_name_hf,
        model_name=run_config.model_name,
        model_load_config=model_load_config,
        lora_config=lora_config,
        use_unsloth=not training_args.full_finetune,
        full_finetune=training_args.full_finetune,
    )
    # avoid cache=True default on HF config.json. unsloth patch
    # lets padding be enabled by default and also removes use_cache=False
    # default in SFTTrainer.compute_loss, removing position_ids padding separation  
    model.config.use_cache = False
    tokenizer = chat_template_func(tokenizer)

    mask_prompt = training_args.mask_prompt_tokens
    dataset = apply_chat_template(tokenizer, dataset, mask_prompt=mask_prompt)
    print(f"Example SFT input:\n{dataset['train'][0]['text']}")
    # The rendered template already contains BOS. Supplying input_ids here prevents
    # TRL's language-model text path from tokenizing with the Llama default and
    # prepending a second BOS.
    dataset = tokenize_sft_dataset(
        tokenizer,
        dataset,
        max_length=model_load_config.max_seq_length,
        num_proc=4,
        mask_prompt=mask_prompt,
    )
    # Unsloth turns padding-free batching off whenever the caller supplies a
    # collator or a processor, so let TRL build the collator from
    # ``completion_only_loss`` below and hand it a plain tokenizer. Gemma 3's
    # processor wrapper also lacks ``pad``, which makes Unsloth swap in the
    # generic Transformers collator that ignores the completion mask; the full
    # processor stays available for chat templating, callbacks, and saving.
    trainer_processing_class = getattr(tokenizer, "tokenizer", tokenizer)

    callbacks = [OffsetAwareWandbCallback()]
    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=run_config.model_name_hf,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals or [],
            pipeline_args=pipeline_args,
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
        processing_class = trainer_processing_class,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        callbacks = callbacks,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = model_load_config.max_seq_length,
            dataset_num_proc = 4,
            packing = False,
            # TRL infers this from prompt/completion columns, which a pre-tokenized
            # dataset does not have, so the mask is only honoured when set here.
            completion_only_loss = mask_prompt,
            **training_args.to_hf_args(output_dir=run_config.output_dir),
        ),
    )

    if disable_loss_kwargs_if_unsupported(trainer):
        print(
            f"[SFT] Model normalizes loss per micro-batch; Trainer will divide by "
            f"gradient_accumulation_steps={trainer.args.gradient_accumulation_steps}"
        )

    remove_default_wandb_callback(trainer)

    print(
        f"[SFT] padding-free batching: "
        f"{padding_free_status(model, trainer.args, training_args.full_finetune)}"
    )
    print(trainer.args.to_dict())

    resume_from_checkpoint = None
    if training_args.resume_from_checkpoint and Path(run_config.output_dir).exists():
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
