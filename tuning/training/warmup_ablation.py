# ABOUTME: Standalone SFT script for warmup_ratio ablation experiments.
# ABOUTME: Runs a single SFT job with a specified warmup_ratio and pass@k evaluation.

import os
import sys
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
    "llama3-1B": 0.75,
    "llama3-3B": 0.6,
    "llama3-8B": 0.6,
    "qwen2-2B": 0.65,
    "qwen2-3B": 0.65,
    "qwen2-7B": 0.55,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Warmup ratio ablation: run SFT with a given warmup_ratio."
    )

    parser.add_argument("--model", required=True, choices=list(MODEL_TO_GPU),
                        help="Base model name")
    parser.add_argument("--warmup-ratio", required=True, type=float,
                        help="Warmup ratio (0.0 to 1.0)")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")

    parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k"])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--task-name", default="gsm8k", choices=["ifeval", "gsm8k"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)

    # Pass@k evaluation
    parser.add_argument("--passk-targets", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                                 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                 0.85, 0.9, 0.95])
    parser.add_argument("--passk-temperature", type=float, default=0.5)
    parser.add_argument("--passk-num-prompts", type=int, default=541)

    return parser.parse_args(argv)


def main():
    args = _parse_args()
    print(args)

    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, ModelLoadConfig, LoraConfig,
        TrainingArgumentsConfig, PassAtKConfig,
    )
    from tuning.training.sft_training import train_model_sft

    set_chat_template(args.model)
    gpu_util = MODEL_TO_GPU[args.model]

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="sft",
        train_size=args.train_size,
    )
    run_config = SFTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
        task_name=args.task_name,
    )

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length

    training_args = TrainingArgumentsConfig()
    training_args.warmup_ratio = args.warmup_ratio
    training_args.num_train_epochs = args.num_epochs
    training_args.eval_steps = args.eval_steps
    training_args.per_device_train_batch_size = args.batch_size
    training_args.gradient_accumulation_steps = args.grad_accum

    passk_config = PassAtKConfig(
        target_pass_at_k=args.passk_targets,
        temperature=args.passk_temperature,
        enabled=True,
        vllm_gpu_memory_utilization=gpu_util,
        use_persistent_vllm=False,
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

    warmup_label = str(args.warmup_ratio).replace(".", "-")
    tags = ["sft", "warmup-ablation", f"warmup-{warmup_label}",
            primary_eval.id]

    with wandb.init(
        name=run_config.run_name,
        project=args.wandb_project,
        job_type="sft",
        tags=tags,
        config={
            "stage": "sft",
            "warmup_ratio": args.warmup_ratio,
            "num_epochs": args.num_epochs,
        },
        settings=wandb.Settings(init_timeout=300),
    ):
        train_model_sft(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            primary_eval=primary_eval,
            monitor_evals=[],
        )


if __name__ == "__main__":
    main()
