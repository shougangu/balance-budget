# ABOUTME: CLI-driven unified SFT+post-training pipeline with optional pass@k and perplexity callbacks.
# ABOUTME: Supports SFT→{DPO,GRPO,KTO} runs from a single command.

import json
import sys
import subprocess
from pathlib import Path

import tuning.config

from tuning.training.pipeline.cli import (
    init_cuda_env, is_worker_mode,
    SBATCH_WORKER_SCRIPT_DEFAULT as SBATCH_WORKER_SCRIPT,
    MODEL_TO_GPU_1, MODEL_TO_GPU_2, MODEL_TO_GPU_3, MODEL_TO_SIMPLERL_TIER,
    parse_early_tuple, effective_eval_seed, _resolve_simplerl_dataset,
    _init_seeds, _parse_args,
)
from tuning.training.pipeline.checkpoint_metadata import (
    load_checkpoints, next_checkpoint, claim_next_checkpoint, mark_completed,
    print_metadata_paths, parse_metadata_from_output,
)
from tuning.training.pipeline.eval_components import (
    _build_eval_components, _sft_ppl_config, _dpo_ppl_config,
    _sft_tags, post_training_tags,
)


if is_worker_mode():
    init_cuda_env()
    if "--run-grpo" not in sys.argv:
        import unsloth  # noqa: F401 - must be imported before trl/transformers/peft


def run_sft(args):
    """Run SFT stage, returning a list of metadata file paths written by callbacks."""
    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, ModelLoadConfig, LoraConfig,
        TrainingArgumentsConfig,
    )
    from tuning.training.sft_training import train_model_sft
    from tuning.utils.gpu import cleanup_gpu

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_1[args.model]

    sft_size = args.sft_data_size if args.sft_data_size is not None else args.train_size
    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="sft",
        train_size=sft_size,
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
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="sft",
        tags=tags,
        config={"stage": "sft", "seed": args.seed, "eval_seed": tuning.config.get_eval_seed()},
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
        cb.metadata_path
        for cb in callbacks
        if getattr(cb, "metadata_path", None)
    ]

    del model, tokenizer, trainer, callbacks
    cleanup_gpu()
    print(subprocess.check_output("nvidia-smi").decode())
    print_metadata_paths(metadata_paths)
    return metadata_paths


def run_dpo(args):
    """Run DPO for the next non-completed checkpoint in the metadata file.

    Processes exactly one checkpoint, marks it completed, then returns.
    Process exit frees all GPU memory.
    """
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)

    fixed_split = args.sft_data_size is not None and args.dpo_data_size is not None
    if fixed_split:
        dpo_size = args.dpo_data_size
    else:
        dpo_size = args.train_size - checkpoint["data_points_seen"]
    if dpo_size <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
        return

    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
        LoraConfig, DPOTrainingConfig,
    )
    from tuning.training.dpo_training import train_model_dpo

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_2[args.model]
    model_name = Path(checkpoint["checkpoint_path"]).name

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="pt",
        train_size=dpo_size,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
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
        do_training=True,
    )
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = DPOTrainingConfig()
    training_args.num_train_epochs = args.dpo_num_epochs
    training_args.eval_steps = args.dpo_eval_steps
    training_args.per_device_train_batch_size = args.dpo_batch_size
    training_args.gradient_accumulation_steps = args.dpo_grad_accum
    training_args.learning_rate = args.dpo_learning_rate

    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "dpo", gpu_util)
    ppl_config = _dpo_ppl_config(args)

    initial_step = checkpoint.get("global_step", 0)
    if passk_config is not None:
        passk_config.initial_global_step = initial_step
    if ppl_config is not None:
        ppl_config.initial_global_step = initial_step

    perplexity_test_dataset = None
    if ppl_config is not None:
        from tuning.data.train_dataset import get_train_dataset
        sft_dataset = get_train_dataset(sft_run_config)
        perplexity_test_dataset = sft_dataset["test"]

    tags = post_training_tags("dpo", checkpoint, primary_eval, passk_config, ppl_config)

    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="dpo",
        tags=tags,
        config={"stage": "dpo", "seed": args.seed, "eval_seed": tuning.config.get_eval_seed()},
        settings=wandb.Settings(init_timeout=300)
    ):
        train_model_dpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            perplexity_test_dataset=perplexity_test_dataset,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=checkpoint.get("global_step")
        )

    mark_completed(metadata_file, checkpoint["checkpoint_path"])


def _build_reward_funcs(args):
    """Build reward function list based on task name."""
    from tuning.training.reward_functions import gsm8k_reward_func, math500_reward_func, ifeval_reward_func
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


def run_grpo(args):
    """Run GRPO for the next non-completed checkpoint in the metadata file.

    Processes exactly one checkpoint, marks it completed, then returns.
    Process exit frees all GPU memory.
    """
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)

    fixed_split = args.sft_data_size is not None and args.grpo_data_size is not None
    if fixed_split:
        grpo_size = args.grpo_data_size
    else:
        grpo_size = args.train_size - checkpoint["data_points_seen"]
    if grpo_size <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
        return

    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
        LoraConfig, GRPOTrainingConfig,
    )
    from tuning.training.grpo_training import train_model_grpo

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_3[args.model]
    model_name = Path(checkpoint["checkpoint_path"]).name

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="rlvr",
        train_size=grpo_size,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
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
        pft_method="grpo",
        do_training=True,
        simple_template=args.simple_template,
    )
    lora_config = LoraConfig()
    if args.grpo_lora_target_modules is not None:
        lora_config.target_modules = args.grpo_lora_target_modules
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = GRPOTrainingConfig()
    training_args.num_train_epochs = args.grpo_num_epochs
    training_args.eval_steps = args.grpo_eval_steps
    training_args.per_device_train_batch_size = args.grpo_batch_size
    training_args.gradient_accumulation_steps = args.grpo_grad_accum
    training_args.num_generations = args.grpo_num_generations
    training_args.max_completion_length = args.grpo_max_completion_length
    # training_args.max_prompt_length = args.grpo_max_prompt_length
    training_args.beta = args.grpo_beta
    training_args.temperature = args.grpo_temperature
    training_args.learning_rate = args.grpo_learning_rate
    training_args.loss_type = args.grpo_loss_type
    scale_rewards = args.grpo_scale_rewards
    training_args.scale_rewards = False if scale_rewards == "false" else scale_rewards
    training_args.vllm_gpu_memory_utilization = gpu_util

    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "grpo", gpu_util)
    reward_funcs = _build_reward_funcs(args)

    initial_step = checkpoint.get("global_step", 0)
    if passk_config is not None:
        passk_config.initial_global_step = initial_step

    tags = post_training_tags("grpo", checkpoint, primary_eval, passk_config, ppl_config=None)

    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="grpo",
        tags=tags,
        config={
            "stage": "grpo",
            "seed": args.seed,
            "eval_seed": tuning.config.get_eval_seed(),
        },
        settings=wandb.Settings(init_timeout=300)
    ):
        train_model_grpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            reward_funcs=reward_funcs,
            passk_config=passk_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=checkpoint.get("global_step"),
            lora_layers_fraction=args.grpo_lora_layers_fraction,
        )

    mark_completed(metadata_file, checkpoint["checkpoint_path"])


def _build_base_cmd(argv):
    """Build base subprocess command by stripping orchestrator-only flags."""
    return [a for a in argv if a != "--run-all"]


def _submit_sbatch_worker(sbatch_script, worker_args):
    """Submit an sbatch worker job, return the Slurm job ID as a string.

    Exits the orchestrator on sbatch error or unparseable output.
    """
    cmd = ["sbatch", sbatch_script, *worker_args]
    print(f"[orchestrator] Submitting sbatch worker: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sbatch failed (code {result.returncode}): {result.stderr.strip()}")
    tokens = result.stdout.strip().split()
    if len(tokens) < 4 or tokens[0] != "Submitted":
        sys.exit(f"Unexpected sbatch stdout: {result.stdout!r}")
    return tokens[-1]


def _dispatch_parallel_workers(parallel, base_cmd, pt_flag, metadata_files):
    """Submit parallel-1 sbatch workers for post-training.

    No-op when parallel <= 1. Strips --parallel from worker args so
    workers don't recursively dispatch.
    """
    if parallel <= 1:
        return

    worker_argv = []
    skip_next = False
    for tok in base_cmd[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--parallel":
            skip_next = True
            continue
        worker_argv.append(tok)
    worker_argv += [pt_flag, "--run-all"]
    for mf in metadata_files:
        if Path(mf).is_file():
            worker_argv += ["--metadata-file", mf]

    for i in range(parallel - 1):
        job_id = _submit_sbatch_worker(SBATCH_WORKER_SCRIPT, worker_argv)
        print(f"[orchestrator] Submitted worker {i+1}/{parallel-1}: job {job_id}")


def main():
    args = _parse_args()
    _resolve_simplerl_dataset(args)
    print(args)

    if not any([args.run_sft, args.run_dpo, args.run_grpo, args.run_all]):
        args.run_all = True

    # Worker mode: run in-process
    if args.run_sft and not args.run_all:
        run_sft(args)
        return

    if args.run_dpo and not args.run_all:
        run_dpo(args)
        return

    if args.run_grpo and not args.run_all:
        run_grpo(args)
        return

    # Orchestrator mode: spawn subprocesses
    base_cmd = _build_base_cmd(sys.argv)
    all_files = (args.metadata_file or [])
    if not (args.run_dpo or args.run_grpo):
        # SFT subprocess
        sft_cmd = [sys.executable] + base_cmd + ["--run-sft"]
        print(f"[orchestrator] Running SFT: {' '.join(sft_cmd)}")
        result = subprocess.run(sft_cmd, stdout=subprocess.PIPE, text=True)
        print(result.stdout)
        if result.returncode != 0:
            sys.exit(f"SFT subprocess failed with return code {result.returncode}")

        metadata_files = parse_metadata_from_output(result.stdout)
        if not metadata_files and not args.metadata_file:
            sys.exit("No metadata files from SFT and no --metadata-file provided")
        all_files = metadata_files + (args.metadata_file or [])

        print(f"Metadata files for post-training: {all_files}")

    # Post-training subprocess loop: one subprocess per checkpoint
    pt_method = args.post_training_method
    pt_flag = f"--run-{pt_method}" if pt_method != "dpo" else "--run-dpo"
    _dispatch_parallel_workers(
        parallel=args.parallel,
        base_cmd=base_cmd,
        pt_flag=pt_flag,
        metadata_files=all_files,
    )
    for metadata_file in all_files:
        if not Path(metadata_file).is_file():
            print(f"Warning: metadata file {metadata_file} does not exist, skipping")
            continue
        while True:
            pt_cmd = [sys.executable] + base_cmd + [
                pt_flag, "--metadata-file", metadata_file,
            ]
            print(f"[orchestrator] Running {pt_method.upper()}: {' '.join(pt_cmd)}")
            result = subprocess.run(pt_cmd)
            if result.returncode == 42:
                print(f"[orchestrator] No more checkpoints in {metadata_file}, moving on")
                break
            if result.returncode != 0:
                sys.exit(f"{pt_method.upper()} subprocess failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
