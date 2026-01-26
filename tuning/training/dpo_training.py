import json
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from tuning.config import MODELS_DIR
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import PTRunConfig, LoraConfig, ModelLoadConfig, DatasetConfig, DPOTrainingConfig, SFTRunConfig, PassAtKConfig, dpo_batch_size, effective_batch_size
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.utils.utils import apply_chat_template_pt, chat_template_func
from tuning.config import MODELS_DIR, DATASETS_DIR
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig
import pprint
from typing import List, Optional, Union

PatchDPOTrainer()

import torch
import gc

def train_model_dpo(
        run_config: PTRunConfig = None,
        lora_config: LoraConfig = None,
        model_load_config: ModelLoadConfig = None,
        training_args: DPOTrainingConfig = None,
        perplexity_thresholds: Optional[List[float]] = None,
        passk_config = None,  # PassAtKConfig object
):

    train_batch_size = dpo_batch_size(run_config.dataset_config.train_size)
    gradient_accumulation_steps = effective_batch_size(run_config.dataset_config.train_size) // train_batch_size

    print(f"Per device train batch size: {train_batch_size}")

    dataset = get_train_dataset(run_config)
    
    if run_config.sft_run_config:
        if run_config.sft_run_config.dataset_config.dynamic_path:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.dataset_config.dynamic_path}"    
        else:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.run_name}"
    else:
        model_path = run_config.model_name_hf

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = model_load_config.max_seq_length,
        dtype = model_load_config.dtype,
        load_in_4bit = model_load_config.load_in_4bit,
    )

    dataset = apply_chat_template_pt(tokenizer, dataset)

    pprint.pprint(dataset["train"][0])
    pprint.pprint(dataset)

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    print(f"Using LORA with config: {lora_config.target_modules}")
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

    print(f"Model loaded - {type(model)}")

    # Print memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()

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
        )
        callbacks.append(passk_callback)
        print(f"[DPO] Will stop training when pass@{passk_config.k_values[0]} >= {passk_config.target_pass_at_k}")

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
        print(f"[DPO] Perplexity thresholds: {perplexity_thresholds}")
        print(f"[DPO] Will checkpoint at each threshold and stop at final: {min(perplexity_thresholds)}")

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        tokenizer = chat_template_func(tokenizer),
        beta = training_args.beta,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        max_length = model_load_config.max_seq_length,
        callbacks = callbacks if callbacks else None,
        args = DPOConfig(
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            gradient_accumulation_steps = training_args.gradient_accumulation_steps,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            warmup_ratio = training_args.warmup_ratio,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            do_eval = training_args.do_eval,
            eval_strategy = training_args.eval_strategy,
            eval_steps = training_args.eval_steps,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            report_to = training_args.report_to,
            logging_steps = training_args.logging_steps,

            output_dir = run_config.output_dir,
            save_strategy = "no",

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = 42,
        )
    )

    trainer_stats = trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method = "merged_16bit")

    args = trainer.args.to_dict()
    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)

    return model, tokenizer, trainer    

if __name__ == "__main__":
    from tuning.config import HF_MODEL_MAP
    
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
            perplexity_thresholds=[0.1],  # Set to a value like 1.5 to enable
            passk_config=passk_config,
        )
        

