# ABOUTME: Stage runners for SFT and post-training (DPO/GRPO). run_post_training is the
# ABOUTME: shared helper; run_dpo and run_grpo are thin wrappers over it.

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import wandb

import tuning.config
from tuning.config import HF_MODEL_MAP, set_chat_template
from tuning.training.config_training import (
    DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
    LoraConfig, TrainingArgumentsConfig, DPOTrainingConfig, GRPOTrainingConfig,
)
from tuning.utils.gpu import cleanup_gpu

from tuning.training.pipeline.checkpoint_metadata import (
    claim_next_checkpoint, mark_completed,
)
from tuning.training.pipeline.cli import (
    MODEL_TO_GPU_1, MODEL_TO_GPU_2, MODEL_TO_GPU_3, _init_seeds,
)
from tuning.training.pipeline.eval_components import (
    _build_eval_components, _sft_ppl_config, _dpo_ppl_config,
    _sft_tags, post_training_tags,
)


def run_sft(args):
    """Run SFT stage, returning a list of metadata file paths written by callbacks."""
    import subprocess
    from tuning.training.sft_training import train_model_sft

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_1[args.model]

    sft_size = args.sft_data_size if args.sft_data_size is not None else args.train_size
    dataset_config = DatasetConfig(
        dataset= args.sft_dataset if args.sft_dataset else args.dataset, 
        dataset_type="sft", 
        train_size=sft_size,
    )
    run_config = SFTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        do_training=True, do_inference=False, do_evaluation=False,
        task_name=args.task_name,
    )
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = TrainingArgumentsConfig()
    training_args.num_train_epochs = args.sft_num_epochs
    training_args.eval_steps = args.sft_eval_steps
    training_args.per_device_train_batch_size = args.sft_batch_size
    training_args.gradient_accumulation_steps = args.sft_grad_accum
    training_args.warmup_ratio = args.sft_warmup_ratio
    training_args.learning_rate = args.sft_learning_rate

    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "sft", gpu_util)
    ppl_config = _sft_ppl_config(args)
    tags = _sft_tags(passk_config, ppl_config, primary_eval)

    with wandb.init(
        name=run_config.model_name, project=args.wandb_project,
        job_type="sft", tags=tags,
        config={"stage": "sft", "seed": args.seed,
                "eval_seed": tuning.config.get_eval_seed()},
        settings=wandb.Settings(init_timeout=300),
    ):
        model, tokenizer, trainer, callbacks = train_model_sft(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
        )

    metadata_paths = [
        cb.metadata_path for cb in callbacks if getattr(cb, "metadata_path", None)
    ]

    del model, tokenizer, trainer, callbacks
    cleanup_gpu()
    print(subprocess.check_output("nvidia-smi").decode())
    from tuning.training.pipeline.checkpoint_metadata import print_metadata_paths
    print_metadata_paths(metadata_paths)
    return metadata_paths


def _build_reward_funcs(args):
    """Build reward function list based on dataset name."""
    from tuning.training.reward_functions import (
        gsm8k_reward_func, math500_reward_func, ifeval_reward_func,
    )
    if args.dataset == "gsm8k":
        return [gsm8k_reward_func]
    elif args.dataset == "openmath":
        return [math500_reward_func]
    elif args.dataset == "ifeval":
        return [ifeval_reward_func]
    elif args.dataset == "ifrlvr":
        from tuning.training.reward_functions import ifrlvr_reward_func
        return [ifrlvr_reward_func]
    elif args.dataset in {"simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"}:
        return [math500_reward_func]
    else:
        raise ValueError(f"No reward function for task: {args.dataset}")


@dataclass
class _PostTrainingConfigs:
    dataset_config: DatasetConfig
    sft_run_config: SFTRunConfig
    run_config: PTRunConfig
    lora_config: LoraConfig
    model_load_config: ModelLoadConfig
    training_args: DPOTrainingConfig | GRPOTrainingConfig
    gpu_util: float


def _resolve_remaining_budget(args, method: str, checkpoint) -> int:
    """Return the data budget for the post-training stage."""
    fixed_size = getattr(args, f"{method}_data_size")
    if fixed_size:
        return fixed_size
    return args.train_size - checkpoint["data_points_seen"]


def _build_post_training_configs(
    args, method: Literal["dpo", "grpo"], checkpoint, train_size: int,
) -> _PostTrainingConfigs:
    """Construct dataclass with all configs needed for the post-training stage."""
    model_name = Path(checkpoint["checkpoint_path"]).name
    gpu_util_map = MODEL_TO_GPU_2 if method == "dpo" else MODEL_TO_GPU_3
    gpu_util = gpu_util_map[args.model]

    dataset_type = "pt" if method == "dpo" else "rlvr"
    dataset_config = DatasetConfig(
        dataset=args.dataset, dataset_type=dataset_type, train_size=train_size,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset, dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
        ),
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        task_name=args.task_name,
    )
    run_config_kwargs = dict(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        sft_run_config=sft_run_config,
        task_name=args.task_name,
        pft_method=method,
        do_training=True,
    )
    if method == "grpo":
        run_config_kwargs["simple_template"] = args.simple_template
    run_config = PTRunConfig(**run_config_kwargs)

    lora_config = LoraConfig()
    if method == "grpo" and args.grpo_lora_target_modules is not None:
        lora_config.target_modules = args.grpo_lora_target_modules

    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length

    if method == "dpo":
        training_args = DPOTrainingConfig()
        training_args.num_train_epochs = args.dpo_num_epochs
        training_args.eval_steps = args.dpo_eval_steps
        training_args.per_device_train_batch_size = args.dpo_batch_size
        training_args.gradient_accumulation_steps = args.dpo_grad_accum
        training_args.learning_rate = args.dpo_learning_rate
    else:
        training_args = GRPOTrainingConfig()
        training_args.num_train_epochs = args.grpo_num_epochs
        training_args.eval_steps = args.grpo_eval_steps
        training_args.per_device_train_batch_size = args.grpo_batch_size
        training_args.gradient_accumulation_steps = args.grpo_grad_accum
        training_args.num_generations = args.grpo_num_generations
        training_args.max_completion_length = args.grpo_max_completion_length
        training_args.beta = args.grpo_beta
        training_args.temperature = args.grpo_temperature
        training_args.learning_rate = args.grpo_learning_rate
        training_args.loss_type = args.grpo_loss_type
        scale_rewards = args.grpo_scale_rewards
        training_args.scale_rewards = False if scale_rewards == "false" else scale_rewards
        training_args.vllm_gpu_memory_utilization = gpu_util

    return _PostTrainingConfigs(
        dataset_config=dataset_config,
        sft_run_config=sft_run_config,
        run_config=run_config,
        lora_config=lora_config,
        model_load_config=model_load_config,
        training_args=training_args,
        gpu_util=gpu_util,
    )


def _train_dispatch(method, configs, passk_config, primary_eval,
                    monitor_evals, ppl_config, checkpoint):
    """Call the right train_model_* with the right kwargs."""
    initial_step = checkpoint.get("global_step", 0)
    if passk_config is not None:
        passk_config.initial_global_step = initial_step
    if ppl_config is not None:
        ppl_config.initial_global_step = initial_step

    if method == "dpo":
        from tuning.training.dpo_training import train_model_dpo

        perplexity_test_dataset = None
        if ppl_config is not None:
            from tuning.data.train_dataset import get_train_dataset
            sft_dataset = get_train_dataset(configs.sft_run_config)
            perplexity_test_dataset = sft_dataset["test"]
        train_model_dpo(
            run_config=configs.run_config,
            lora_config=configs.lora_config,
            model_load_config=configs.model_load_config,
            training_args=configs.training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            perplexity_test_dataset=perplexity_test_dataset,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=initial_step,
        )
    else:
        from tuning.training.grpo_training import train_model_grpo

        train_model_grpo(
            run_config=configs.run_config,
            lora_config=configs.lora_config,
            model_load_config=configs.model_load_config,
            training_args=configs.training_args,
            reward_funcs=_build_reward_funcs(_train_dispatch._args),
            passk_config=passk_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=initial_step,
            lora_layers_fraction=_train_dispatch._args.grpo_lora_layers_fraction,
        )


def run_post_training(args, method: Literal["dpo", "grpo"]):
    """Claim → check budget → build configs → wandb run → train → mark completed."""
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)

    train_size = _resolve_remaining_budget(args, method, checkpoint)
    if train_size <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
        return

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)

    configs = _build_post_training_configs(args, method, checkpoint, train_size)
    passk_config, primary_eval, monitor_evals = _build_eval_components(
        args, method, configs.gpu_util,
    )
    ppl_config = _dpo_ppl_config(args) if method == "dpo" else None
    tags = post_training_tags(method, checkpoint, primary_eval, passk_config, ppl_config)

    _train_dispatch._args = args
    try:
        with wandb.init(
            name=configs.run_config.model_name,
            project=args.wandb_project,
            job_type=method, tags=tags,
            config={"stage": method, "seed": args.seed,
                    "eval_seed": tuning.config.get_eval_seed()},
            settings=wandb.Settings(init_timeout=300),
        ):
            _train_dispatch(method, configs, passk_config, primary_eval,
                            monitor_evals, ppl_config, checkpoint)
    finally:
        del _train_dispatch._args

    mark_completed(metadata_file, checkpoint["checkpoint_path"])


def run_dpo(args):
    return run_post_training(args, "dpo")


def run_grpo(args):
    return run_post_training(args, "grpo")
