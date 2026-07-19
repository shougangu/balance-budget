# ABOUTME: Stage runners for SFT and post-training (DPO/GRPO). run_post_training is the
# ABOUTME: shared helper; run_dpo and run_grpo are thin wrappers over it.

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import wandb

import tuning.config
from tuning.config import HF_MODEL_MAP, set_chat_template
from tuning.training.config_training import (
    DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
    LoraConfig, TrainingArgumentsConfig, DPOTrainingConfig, GRPOTrainingConfig,
)
from tuning.utils.gpu import cleanup_gpu

from tuning.training.pipeline.checkpoint_metadata import (
    claim_checkpoint, claim_next_checkpoint, mark_completed, record_wandb_run_id,
)
from tuning.training.pipeline.cli import (
    MODEL_TO_GPU_1, MODEL_TO_GPU_2, MODEL_TO_GPU_3, _init_seeds,
)
from tuning.training.pipeline.eval_components import (
    _build_eval_components, _sft_ppl_config, _dpo_ppl_config,
    _sft_tags, post_training_tags,
)


def _build_lora_config(args) -> LoraConfig:
    """LoraConfig with CLI-controllable knobs applied (gradient checkpointing, etc.)."""
    cfg = LoraConfig()
    cfg.r = args.lora_rank
    cfg.lora_alpha = round(32 * (args.lora_rank / 32) ** 0.5)  # rsLoRA-equivalent alpha/√r scaling
    if not getattr(args, "gradient_checkpointing", True):
        cfg.use_gradient_checkpointing = False
    return cfg


def run_sft(args):
    """Run SFT stage, returning a list of metadata file paths written by callbacks."""
    import subprocess
    from tuning.training.sft_training import train_model_sft

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_1[args.model]
    if args.sft_full_finetune:
        # if "paged" in args.sft_optim:
        #     # Ephemeral eval moves both model weights and managed optimizer
        #     # pages off GPU before vLLM performs its startup memory check.
        #     gpu_util = 0.9
        # else:
        #     # An explicit non-paged optimizer override cannot be migrated by
        #     # the eval runner, so leave room for its GPU-resident state.
        gpu_util = min(gpu_util, 0.56)

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
    lora_config = _build_lora_config(args)
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = TrainingArgumentsConfig()
    training_args.resume_from_checkpoint = bool(args.sft_resume)
    training_args.num_train_epochs = args.sft_num_epochs
    training_args.eval_steps = args.sft_eval_steps
    training_args.per_device_train_batch_size = args.sft_batch_size
    training_args.gradient_accumulation_steps = args.sft_grad_accum
    training_args.warmup_ratio = args.sft_warmup_ratio
    training_args.lr_scheduler_type = args.sft_lr_scheduler_type
    training_args.learning_rate = args.sft_learning_rate
    training_args.full_finetune = args.sft_full_finetune
    training_args.optim = args.sft_optim

    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "sft", gpu_util)
    ppl_config = _sft_ppl_config(args)
    tags = _sft_tags(passk_config, ppl_config, primary_eval) + args.tags
    if args.sft_full_finetune:
        tags = tags + ["fullft"]

    wandb_ctx = _init_wandb_run(args, run_config.model_name, "sft", tags, args.sft_resume)
    run_config.wandb_run_id = wandb_ctx.id if wandb_ctx else ""
    with wandb_ctx:
        model, tokenizer, trainer, callbacks = train_model_sft(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            pipeline_args=args,
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
    elif args.dataset in {"dapo", "mathmix", "simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"}:
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
    gpu_util = (
        args.grpo_gpu_util if method == "grpo" and args.grpo_gpu_util is not None
        else gpu_util_map[args.model]
    )

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

    lora_config = _build_lora_config(args)
    if method == "grpo" and args.grpo_lora_target_modules is not None:
        lora_config.target_modules = args.grpo_lora_target_modules

    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length

    if method == "grpo" and args.grpo_precision == "fp16":
        model_load_config.dtype = torch.float16
    elif method == "grpo" and args.grpo_precision == "bf16":
        model_load_config.dtype = torch.bfloat16

    if method == "dpo":
        training_args = DPOTrainingConfig()
        training_args.num_train_epochs = args.dpo_num_epochs
        training_args.eval_steps = args.dpo_eval_steps
        training_args.per_device_train_batch_size = args.dpo_batch_size
        training_args.gradient_accumulation_steps = args.dpo_grad_accum
        training_args.learning_rate = args.dpo_learning_rate
    else:
        training_args = GRPOTrainingConfig()
        training_args.num_compute_gpus = args.grpo_num_gpus
        training_args.num_train_epochs = args.grpo_num_epochs
        training_args.eval_steps = args.grpo_eval_steps
        training_args.per_device_train_batch_size = args.grpo_batch_size
        if args.grpo_eval_batch_size is not None:
            training_args.per_device_eval_batch_size = args.grpo_eval_batch_size
        training_args.gradient_accumulation_steps = args.grpo_grad_accum
        training_args.num_generations = args.grpo_num_generations
        training_args.num_iterations = args.grpo_num_iterations
        training_args.max_completion_length = args.grpo_max_completion_length
        training_args.vllm_max_model_length = 4096 + args.grpo_max_completion_length
        training_args.beta = args.grpo_beta
        training_args.temperature = args.grpo_temperature
        training_args.learning_rate = args.grpo_learning_rate
        training_args.warmup_ratio = args.grpo_warmup_ratio
        training_args.lr_scheduler_type = args.grpo_lr_scheduler_type
        training_args.loss_type = args.grpo_loss_type
        if training_args.loss_type == "cispo":
            training_args.epsilon_high = 3.0
            training_args.num_iterations = 4
        scale_rewards = args.grpo_scale_rewards
        training_args.scale_rewards = False if scale_rewards == "false" else scale_rewards
        training_args.vllm_gpu_memory_utilization = gpu_util
        training_args.vllm_enable_sleep_mode = args.grpo_vllm_sleep_mode
        training_args.vllm_importance_sampling_correction = args.grpo_vllm_importance_sampling
        training_args.upcast_lm_head_fp32 = args.grpo_upcast_lm_head_fp32
        training_args.precision = args.grpo_precision
        training_args.use_liger_kernel = args.grpo_use_liger_kernel
        training_args.zero_variance_filter = args.grpo_zero_variance_filter
        training_args.zero_variance_filter_epsilon = args.grpo_zero_variance_filter_epsilon
        training_args.vllm_mode = args.grpo_vllm_mode
        if args.grpo_vllm_mode == "server":
            training_args.dataloader_num_workers = 0
        training_args.vllm_server_host = args.grpo_vllm_server_host
        training_args.vllm_server_port = args.grpo_vllm_server_port
        training_args.vllm_group_port = args.grpo_vllm_group_port
        training_args.vllm_server_timeout = args.grpo_vllm_server_timeout

    training_args.resume_from_checkpoint = bool(checkpoint.get("continue", False))

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


def _init_wandb_run(args, run_name, job_type, tags, wandb_run_id: str = ""):
    """Open a wandb run; resume an existing run when wandb_run_id is provided."""
    kwargs = dict(
        name=run_name,
        project=args.wandb_project,
        job_type=job_type, tags=tags,
        config={"stage": job_type, "seed": args.seed,
                "eval_seed": tuning.config.get_eval_seed()},
        settings=wandb.Settings(init_timeout=300),
    )
    if wandb_run_id:
        kwargs["id"] = wandb_run_id
        kwargs["resume"] = "must"
    run = wandb.init(**kwargs)
    if not wandb_run_id:
        wandb.alert(
            title=f"Run started: {run_name}",
            text=f"job_type={job_type} tags={tags}\n\n{' '.join(sys.argv)}",
        )
    return run


def run_post_training(args, method: Literal["dpo", "grpo"]):
    """Claim → check budget → build configs → wandb run → train → mark completed."""
    metadata_file = args.metadata_file[0]

    # Bring up the process group early so claim/broadcast/barrier work before
    # the trainer is constructed. HF Accelerator detects an existing group and
    # reuses it.
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_dist = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if is_dist else 0

    pinned = getattr(args, "claim_checkpoint", None)
    if rank == 0:
        checkpoint = (claim_checkpoint(metadata_file, pinned) if pinned
                      else claim_next_checkpoint(metadata_file))
    else:
        checkpoint = None
    if is_dist:
        payload = [checkpoint]
        dist.broadcast_object_list(payload, src=0)
        checkpoint = payload[0]

    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)

    train_size = _resolve_remaining_budget(args, method, checkpoint)
    if train_size <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        if rank == 0:
            mark_completed(metadata_file, checkpoint["checkpoint_path"])
        if is_dist:
            dist.barrier()
        return

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)

    if rank == 0:
        from tuning.utils.utils import warn_if_template_mismatch
        warn_if_template_mismatch(checkpoint["checkpoint_path"], args.simple_template)

    configs = _build_post_training_configs(args, method, checkpoint, train_size)
    passk_config, primary_eval, monitor_evals = _build_eval_components(
        args, method, configs.gpu_util,
    )
    ppl_config = _dpo_ppl_config(args) if method == "dpo" else None
    tags = post_training_tags(method, checkpoint, primary_eval, passk_config, ppl_config) + args.tags

    _train_dispatch._args = args
    try:
        if rank == 0:
            # Only resume the W&B run when continuing an in-progress post-training run.
            # On fresh forks from an SFT checkpoint, multiple post-training runs may share
            # the same SFT checkpoint, so each needs its own W&B run.
            resume_id = checkpoint.get("wandb_run_id", "") if checkpoint.get("continue") else ""
            wandb_ctx = _init_wandb_run(
                args, configs.run_config.model_name, method, tags, resume_id,
            )
            wandb_run_id = wandb_ctx.id if wandb_ctx else ""
            record_wandb_run_id(metadata_file, checkpoint["checkpoint_path"], wandb_run_id)
        else:
            wandb_ctx = contextlib.nullcontext()
            wandb_run_id = ""
        if is_dist:
            payload = [wandb_run_id]
            dist.broadcast_object_list(payload, src=0)
            wandb_run_id = payload[0]
        configs.run_config.wandb_run_id = wandb_run_id
        with wandb_ctx:
            _train_dispatch(method, configs, passk_config, primary_eval,
                            monitor_evals, ppl_config, checkpoint)
    finally:
        del _train_dispatch._args

    if rank == 0:
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
    if is_dist:
        dist.barrier()


def run_dpo(args):
    return run_post_training(args, "dpo")


def run_grpo(args):
    return run_post_training(args, "grpo")
