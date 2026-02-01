from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, PassAtKConfig, sft_batch_size, effective_batch_size
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

def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
    perplexity_thresholds: Optional[List[float]] = None,
    perplexity_window: Optional[int] = None,
    passk_config = None,  # PassAtKConfig object
):  
    train_batch_size = sft_batch_size(run_config.dataset_config.train_size)
    gradient_accumulation_steps = effective_batch_size(run_config.dataset_config.train_size) // train_batch_size

    dataset = get_train_dataset(run_config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = run_config.model_name_hf,
        max_seq_length = model_load_config.max_seq_length,
        dtype = model_load_config.dtype,
        load_in_4bit = model_load_config.load_in_4bit,
    )

    dataset = apply_chat_template(tokenizer, dataset)
    
    print(dataset)  
    print(dataset["train"][0])

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    
    model = FastLanguageModel.get_peft_model(
        model,  
        r = lora_config.r,
        target_modules = lora_config.target_modules,
        lora_alpha = lora_config.lora_alpha, 
        lora_dropout = lora_config.lora_dropout,
        bias = lora_config.bias,
        use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
        random_state = lora_config.random_state, 
        use_rslora = lora_config.use_rslora,
        loftq_config = lora_config.loftq_config,
    )

    # Setup callbacks
    callbacks = []
    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            target_pass_at_k=passk_config.target_pass_at_k,  
            tokenizer=chat_template_func(tokenizer),
            k_values=passk_config.k_values,
            n_samples=passk_config.n_samples,
            num_prompts=passk_config.num_prompts,
            temperature=passk_config.temperature,
            max_tokens=passk_config.max_tokens,
            strict=passk_config.strict,
            model_name=run_config.model_name,
            
        )
        callbacks.append(passk_callback)
        print(f"[SFT] Will stop training when pass@{passk_config.k_values[0]} >= {passk_config.target_pass_at_k[-1]}")
        print(f"[SFT] Checkpoints will be saved at thresholds: {passk_config.target_pass_at_k}")

    if perplexity_thresholds is not None:
        # Load raw dataset (without chat template applied) for perplexity evaluation
        raw_test_dataset = load_from_disk(f"{DATASETS_DIR}/sft-tuluif")["test"]
        raw_test_dataset = dataset["test"]
        perplexity_callback = PerplexityStoppingCallback(
            model_name = run_config.model_name,
            perplexity_thresholds=perplexity_thresholds,
            test_dataset=raw_test_dataset,
            tokenizer=chat_template_func(tokenizer),
            num_samples=200,  
        )
        callbacks.append(perplexity_callback)
        print(f"[SFT] Perplexity thresholds: {perplexity_thresholds}")
        print(f"[SFT] Will checkpoint at each threshold and stop at final: {min(perplexity_thresholds)}")

   

    trainer = SFTTrainer(
        model = model,
        tokenizer = chat_template_func(tokenizer),
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = model_load_config.max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        callbacks = callbacks if callbacks else None,
        args = TrainingArguments(
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            gradient_accumulation_steps = training_args.gradient_accumulation_steps,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            eval_steps = training_args.eval_steps,
            do_eval = training_args.do_eval,
            eval_strategy = training_args.eval_strategy,
            warmup_ratio = training_args.warmup_ratio,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            report_to = training_args.report_to,
            logging_steps = training_args.logging_steps,
            output_dir = run_config.output_dir,
            save_strategy = training_args.save_strategy,
            save_steps = training_args.save_steps,
            save_total_limit = training_args.save_total_limit,
            load_best_model_at_end = training_args.load_best_model_at_end,
            dataloader_drop_last = training_args.dataloader_drop_last,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = 42,
        ),
    )
    args = trainer.args.to_dict()

    print(args)
    
    # Resume from checkpoint if it exists
    resume_from_checkpoint = None
    if Path(run_config.output_dir).exists():
        checkpoints = list(Path(run_config.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
            print(f"[SFT] Resuming from checkpoint: {resume_from_checkpoint}")
    
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint) 
    # weights, opt, scheduler, step counter

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method = "merged_16bit")

    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)

    return model, tokenizer, trainer, callbacks



if __name__ == "__main__":
    
    model = "llama3-8B"
    # model = "llama3-3B"
    # model = "llama3-1B"
    # model = "qwen2-7B"

    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "sft",
        train_size = 8192, # 29980
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
    # training_args.save_total_limit = 10
    # training_args.load_best_model_at_end = False
    # training_args.dataloader_drop_last = False


    # Configure pass@k evaluation
    # passk_config = PassAtKConfig(
    #     target_pass_at_k=[0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9],
    #     k_values=[1], 
    #     n_samples=64,
    #     num_prompts=75,
    #     temperature=0.7,
    #     strict=True,
    #     enabled=True,
    # )


    perplexity_thresholds = [7.0,6.0, 5.75, 5.5, 5.25, 5.0, 4.75, 4.5, 4.25,4.0, 3.9, 3.8, 3.7, 3.6,3.55,3.5,3.45,3.4,3.35,3.3, 3.25, 3.2, 3.15, 3.1]
    model, tokenizer, trainer = train_model_sft(
        run_config = run_config,
        lora_config = lora_config,
        model_load_config = model_load_config,
        training_args = training_args,
        perplexity_thresholds = perplexity_thresholds, 
    )
    metadata_file = trainer.callbacks[-1].metadata_path
    print(f"SFT training complete. Metadata file: {metadata_file}")




