# ABOUTME: Standalone DPO training script for beta ablation experiments.
# ABOUTME: Runs a single DPO job from an existing SFT checkpoint with a specified beta value.

import os
import sys
import re
import argparse

def _init_cuda_env():
    """Restrict training to GPU 0 and save full GPU list for inference workers."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES_ALL"] = all_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus.split(",")[0]


if __name__ == "__main__":
    _init_cuda_env()
    import unsloth  # noqa: F401 - must be imported before trl/transformers/peft


MODEL_TO_GPU = {
    "llama3-1B": 0.7,
    "llama3-3B": 0.75,
    "llama3-8B": 0.62,
    "qwen2-2B": 0.65,
    "qwen2-3B": 0.65,
    "qwen2-7B": 0.5,
}


def parse_data_points_from_checkpoint(checkpoint_name):
    """Extract data_points_seen from a checkpoint directory name like 'llama3-3B_p@1-0.35_sft-1024'."""
    match = re.search(r'sft-(\d+)', checkpoint_name)
    if not match:
        raise ValueError(f"Cannot extract sft-N from checkpoint name: {checkpoint_name}")
    return int(match.group(1))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DPO beta ablation: run DPO from an SFT checkpoint with a given beta.")

    parser.add_argument("--model", required=True, choices=list(MODEL_TO_GPU),
                        help="Base model name")
    parser.add_argument("--checkpoint", required=True,
                        help="SFT checkpoint directory name (under tuning/models/)")
    parser.add_argument("--beta", required=True, type=float,
                        help="DPO beta value")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")

    parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k"])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--task-name", default="gsm8k", choices=["ifeval", "gsm8k"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)

    # Pass@k evaluation
    parser.add_argument("--passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--passk-temperature", type=float, default=0.5)
    parser.add_argument("--passk-num-prompts", type=int, default=10000)

    return parser.parse_args(argv)


def main():
    args = _parse_args()
    print(args)

    import wandb
    from tuning.config import HF_MODEL_MAP, MODELS_DIR, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
        LoraConfig, DPOTrainingConfig, PassAtKConfig,
    )
    from tuning.training.dpo_training import train_model_dpo

    set_chat_template(args.model)

    data_points_seen = parse_data_points_from_checkpoint(args.checkpoint)
    remaining = args.train_size - data_points_seen
    if remaining <= 0:
        sys.exit(f"No data budget remaining: train_size={args.train_size}, data_points_seen={data_points_seen}")

    checkpoint_path = os.path.join(MODELS_DIR, args.checkpoint)
    if not os.path.isdir(checkpoint_path):
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    gpu_util = MODEL_TO_GPU[args.model]

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="pt",
        train_size=remaining,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=data_points_seen,
            dynamic_path=args.checkpoint,
        ),
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        task_name=args.task_name,
    )
    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        sft_run_config=sft_run_config,
        task_name=args.task_name,
        pft_method="dpo",
        add_beta_run_name=True,
        beta=args.beta,
        do_training=True,
    )

    lora_config = LoraConfig()
    lora_config.use_gradient_checkpointing = True
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length

    training_args = DPOTrainingConfig()
    training_args.beta = args.beta
    training_args.eval_steps = args.eval_steps
    training_args.per_device_train_batch_size = args.batch_size
    training_args.gradient_accumulation_steps = args.grad_accum

    # Pass@k evaluation during DPO
    passk_config = PassAtKConfig(
        target_pass_at_k=args.passk_targets,
        temperature=args.passk_temperature,
        enabled=True,
        vllm_gpu_memory_utilization=gpu_util,
        use_persistent_vllm = False
    )

    if args.task_name == "gsm8k":
        from tuning.training.eval_strategy import GSM8KEvalStrategy
        primary_eval = GSM8KEvalStrategy(
            k_values=[1], n_samples=1, num_prompts=args.passk_num_prompts,
        )
    elif args.task_name == "ifeval":
        from tuning.training.eval_strategy import IFEvalStrategy
        primary_eval = IFEvalStrategy(
            k_values=[1], n_samples=1, num_prompts=args.passk_num_prompts,
        )
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")
    monitor_evals = []

    beta_label = str(args.beta).replace(".", "-")
    tags = ["dpo", "beta-ablation", f"beta-{beta_label}", args.checkpoint]

    with wandb.init(
        name=run_config.run_name,
        project=args.wandb_project,
        job_type="dpo",
        tags=tags,
        config={
            "stage": "dpo",
            "beta": args.beta,
            "sft_checkpoint": args.checkpoint,
            "data_points_seen": data_points_seen,
            "remaining_budget": remaining,
        },
        settings=wandb.Settings(init_timeout=300),
    ):
        train_model_dpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
        )


if __name__ == "__main__":
    main()
