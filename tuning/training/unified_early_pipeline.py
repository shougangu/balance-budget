# ABOUTME: CLI-driven unified SFT+post-training pipeline with optional pass@k and perplexity callbacks.
# ABOUTME: Supports SFT→{DPO,GRPO,KTO} runs from a single command.

import os
import argparse
import json
import sys
import subprocess
from pathlib import Path


def _init_cuda_env():
    """Restrict training to GPU 0 and save full GPU list for inference workers."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES_ALL"] = all_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus.split(",")[0]


def _is_worker_mode():
    """True when running as a training worker (needs CUDA), not as orchestrator or in tests."""
    return ("--run-sft" in sys.argv or "--run-dpo" in sys.argv or "--run-grpo" in sys.argv) and "--run-all" not in sys.argv


if _is_worker_mode():
    _init_cuda_env()
    if not ("--run-grpo" in sys.argv):      
        import unsloth  # noqa: F401 - must be imported before trl/transformers/peft




MODEL_TO_GPU_1 = {
"llama3-1B": 0.75,
"llama3-3B": 0.6, # (0.65 gives 79% peak with multi with one 97% spike?)
"llama3-8B": 0.6,  # (0.68 gives 90% peak)
"qwen2-2B": 0.65,
"qwen2-3B": 0.65, # (0.65 gives 76% peak with non-persistence but one 91% spike?)
"qwen2-7B": 0.55,
}

MODEL_TO_GPU_2 = {
    "llama3-1B": 0.7,
    "llama3-3B": 0.62,  # can reach
    "llama3-8B": 0.62,
    "qwen2-2B": 0.65,
    "qwen2-3B": 0.65,
    "qwen2-7B": 0.5,
}

MODEL_TO_GPU_3 = {
    "llama3-1B": 0.7, # good
    "llama3-3B": 0.65,
    "llama3-8B": 0.5,
    "qwen2-2B": 0.7,
    "qwen2-3B": 0.65, #good
    "qwen2-7B": 0.53,
}


def parse_early_tuple(s):
    """Parse 'patience:min_delta' string into (int, float) tuple."""
    try:
        patience_str, delta_str = s.split(":")
        return (int(patience_str), float(delta_str))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            f"Invalid early tuple {s!r}: expected 'patience:min_delta' (e.g. '2:0.02')"
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified SFT+DPO pipeline with pass@k and perplexity callbacks."
    )

    # Required
    parser.add_argument("--model", required=True, choices=list(MODEL_TO_GPU_1),
                        help="Base model name")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")

    # Stage control
    stage = parser.add_argument_group("stage control")
    stage.add_argument("--run-sft", action="store_true", default=False,
                       help="Run SFT stage only")
    stage.add_argument("--run-dpo", action="store_true", default=False,
                       help="Run DPO stage only")
    stage.add_argument("--run-grpo", action="store_true", default=False,
                       help="Run GRPO stage only")
    stage.add_argument("--run-all", action="store_true", default=False,
                       help="Explicitly run both stages (default when none specified)")
    stage.add_argument("--post-training-method", default="dpo",
                       choices=["dpo", "grpo", "kto"],
                       help="Post-training method to use after SFT")

    # Core
    parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k", "openmath"],)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--sft-data-size", type=int, default=None,
                        help="Fixed SFT data size. When both --sft-data-size and --dpo-data-size are set, "
                             "DPO uses a fixed dataset instead of remaining budget.")
    parser.add_argument("--dpo-data-size", type=int, default=None,
                        help="Fixed DPO data size. When both --sft-data-size and --dpo-data-size are set, "
                             "DPO uses a fixed dataset instead of remaining budget.")
    parser.add_argument("--task-name", default="gsm8k", choices=["ifeval", "gsm8k", "math500"])
    parser.add_argument("--monitor-evals", nargs="*", default=[],
                        choices=["ifeval", "gsm8k", "math500"],
                        help="Additional eval benchmarks to monitor (logged to W&B, no stopping)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-merge", choices=["union", "passk", "ppl"], default="union",
                        help="Which checkpoint types to use for DPO")
    parser.add_argument("--metadata-file", action="append", dest="metadata_file",
                        metavar="FILE",
                        help="Metadata JSONL file from a previous SFT run (repeatable)")

    # SFT-specific
    parser.add_argument("--sft-learning-rate", type=float, default=5e-5)
    parser.add_argument("--sft-warmup-ratio", type=float, default=0.0)

    # Training args
    parser.add_argument("--sft-num-epochs", type=int, default=1)
    parser.add_argument("--sft-eval-steps", type=int, default=64)
    parser.add_argument("--sft-batch-size", type=int, default=16)
    parser.add_argument("--sft-grad-accum", type=int, default=1)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-num-epochs", type=int, default=3)
    parser.add_argument("--dpo-eval-steps", type=int, default=256)
    parser.add_argument("--dpo-batch-size", type=int, default=4)
    parser.add_argument("--dpo-grad-accum", type=int, default=4)
    parser.add_argument("--grpo-num-epochs", type=int, default=1)
    parser.add_argument("--grpo-eval-steps", type=int, default=128)
    parser.add_argument("--grpo-batch-size", type=int, default=8)
    parser.add_argument("--grpo-grad-accum", type=int, default=4)
    parser.add_argument("--grpo-num-generations", type=int, default=8)
    parser.add_argument("--grpo-max-completion-length", type=int, default=1024)
    parser.add_argument("--grpo-max-prompt-length", type=int, default=512)
    parser.add_argument("--grpo-beta", type=float, default=0.0)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-learning-rate", type=float, default=1e-4)
    parser.add_argument("--grpo-loss-type", default="dapo",
                        choices=["grpo", "dr_grpo", "dapo", "bnpo"])
    parser.add_argument("--grpo-scale-rewards", default="group",
                        choices=["group", "batch", "false"],
                        help="Reward scaling method. 'false' disables scaling.")
    parser.add_argument("--grpo-data-size", type=int, default=None,
                        help="Fixed GRPO data size. When set with --sft-data-size, "
                             "GRPO uses a fixed dataset instead of remaining budget.")
    parser.add_argument("--grpo-simple-template", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Skip chat template and pass raw prompt content as plain strings.")

    # GRPO LoRA
    parser.add_argument("--grpo-lora-target-modules", type=str, nargs="+",
                        default=None,
                        help="LoRA target modules (default: all 7 projections)")
    parser.add_argument("--grpo-lora-layers-fraction", type=float, default=1.0,
                        help="Fraction of top layers to apply LoRA to (0.0-1.0, default: 1.0 = all layers)")

    # Callback toggles
    parser.add_argument("--sft-enable-passk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft-enable-ppl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dpo-enable-passk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpo-enable-ppl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-enable-passk", action=argparse.BooleanOptionalAction, default=True)

    # SFT pass@k
    parser.add_argument("--sft-passk-targets", type=float, nargs="+", 
                        default=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
    parser.add_argument("--sft-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--sft-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--sft-passk-n-samples", type=int, default=1)
    parser.add_argument("--sft-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--sft-passk-temperature", type=float, default=0.5)
    parser.add_argument("--sft-passk-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sft-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--sft-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sft-passk-max-checkpoint-gap", type=int, default=None,
                        help="Save a fallback SFT checkpoint if no pass@k checkpoint for this many data points")

    # SFT perplexity
    parser.add_argument("--sft-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--sft-ppl-num-samples", type=int, default=541)
    parser.add_argument("--sft-ppl-early", type=parse_early_tuple, nargs="*",
                        default=[])

    # DPO pass@k
    parser.add_argument("--dpo-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--dpo-passk-early", type=parse_early_tuple, nargs="*", default=[])
    parser.add_argument("--dpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--dpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--dpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--dpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--dpo-passk-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--dpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    # DPO perplexity
    parser.add_argument("--dpo-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dpo-ppl-num-samples", type=int, default=541)
    parser.add_argument("--dpo-ppl-early", type=parse_early_tuple, nargs="*", default=[])

    # GRPO pass@k
    parser.add_argument("--grpo-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--grpo-passk-early", type=parse_early_tuple, nargs="*", default=[])
    parser.add_argument("--grpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--grpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--grpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--grpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--grpo-passk-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--grpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args(argv)
    if (args.sft_enable_ppl or args.dpo_enable_ppl) and args.max_seq_length == 1024:
        args.max_seq_length = 4096
    return args



def _sft_ppl_config(args):
    """Return PerplexityConfig for SFT, or None if disabled."""
    if not args.sft_enable_ppl:
        return None
    from tuning.training.config_training import PerplexityConfig
    return PerplexityConfig(
        perplexity_thresholds=args.sft_ppl_thresholds,
        num_samples=args.sft_ppl_num_samples,
        early_tuples=args.sft_ppl_early or None,
        enabled=True,
    )



def _dpo_ppl_config(args):
    """Return PerplexityConfig for DPO, or None if disabled."""
    if not args.dpo_enable_ppl:
        return None
    from tuning.training.config_training import PerplexityConfig
    return PerplexityConfig(
        perplexity_thresholds=args.dpo_ppl_thresholds,
        num_samples=args.dpo_ppl_num_samples,
        early_tuples=args.dpo_ppl_early or None,
        enabled=True,
    )


def _build_eval_components(args, stage, gpu_util):
    """Build PassAtKConfig + eval strategies for the given task and stage.

    Returns (passk_config, primary_eval, monitor_evals).
    All three are None/[] if pass@k is disabled for this stage.
    """
    prefix = stage  # "sft" or "dpo"
    if not getattr(args, f"{prefix}_enable_passk", False):
        return None, None, []

    from tuning.training.config_training import PassAtKConfig
    passk_config = PassAtKConfig(
        target_pass_at_k=getattr(args, f"{prefix}_passk_targets"),
        early_tuples=getattr(args, f"{prefix}_passk_early") or None,
        temperature=getattr(args, f"{prefix}_passk_temperature"),
        enabled=True,
        num_inference_gpus=getattr(args, f"{prefix}_passk_num_inference_gpus"),
        use_persistent_vllm=getattr(args, f"{prefix}_passk_persistent_vllm"),
        vllm_gpu_memory_utilization=gpu_util,
        max_checkpoint_gap=getattr(args, f"{prefix}_passk_max_checkpoint_gap", None),
    )

    k_values = getattr(args, f"{prefix}_passk_k_values", [1])
    n_samples = getattr(args, f"{prefix}_passk_n_samples", 1)
    num_prompts = getattr(args, f"{prefix}_passk_num_prompts", None)

    if args.task_name == "ifeval":
        from tuning.training.eval_strategy import IFEvalStrategy
        strict = getattr(args, f"{prefix}_passk_strict", True)
        primary_eval = IFEvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts or 541, strict=strict,
        )
    elif args.task_name == "gsm8k":
        from tuning.training.eval_strategy import GSM8KEvalStrategy
        primary_eval = GSM8KEvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts,
        )
    elif args.task_name == "math500":
        from tuning.training.eval_strategy import MATH500EvalStrategy
        primary_eval = MATH500EvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts,
        )
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

    monitor_evals = _build_monitor_evals(args, k_values, n_samples)

    return passk_config, primary_eval, monitor_evals


def _build_monitor_evals(args, k_values, n_samples):
    """Build monitor eval strategies from --monitor-evals arg."""
    monitor_evals = []
    for name in getattr(args, "monitor_evals", []):
        if name == args.task_name:
            continue  # skip if same as primary
        if name == "math500":
            from tuning.training.eval_strategy import MATH500EvalStrategy
            monitor_evals.append(MATH500EvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "gsm8k":
            from tuning.training.eval_strategy import GSM8KEvalStrategy
            monitor_evals.append(GSM8KEvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "ifeval":
            from tuning.training.eval_strategy import IFEvalStrategy
            monitor_evals.append(IFEvalStrategy(k_values=k_values, n_samples=n_samples))
    return monitor_evals


def _sft_tags(passk_config, ppl_config, primary_eval=None):
    """Build W&B tags for an SFT run."""
    from tuning.training.wandb_utils import get_early_pairs, early_pair_tag, get_early_abs, early_abs_tag
    tags = ["sft"]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
        tags.append(early_pair_tag(get_early_pairs(passk_config)))
        # tags.append(early_abs_tag(get_early_abs(passk_config)))
    if ppl_config is not None:
        tags.append("ppl")
        tags.append(early_pair_tag(get_early_pairs(ppl_config)))
        tags.append(early_abs_tag(get_early_abs(ppl_config)))
    if passk_config is None and ppl_config is None:
        tags.append("no_callbacks")
    return tags


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

    set_chat_template(args.model)
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
        config={"stage": "sft"},
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


def load_checkpoints(metadata_files, merge):
    """Load and filter checkpoint rows from one or more JSONL metadata files.

    Args:
        metadata_files: List of file paths to read.
        merge: "union" keeps all rows; "passk" keeps only pass_at_* rows;
               "ppl" keeps only perplexity rows.

    Returns:
        Deduplicated list of checkpoint dicts (first occurrence wins).
    """
    checkpoints = []
    seen_paths = set()
    for path in metadata_files:
        if not Path(path).is_file():
            print(f"Warning: metadata file {path} does not exist, skipping")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                ttype = row.get("threshold_type", "")
                if merge == "passk" and not ttype.startswith("pass_at_"):
                    continue
                if merge == "ppl" and ttype != "perplexity":
                    continue
                cp = row["checkpoint_path"]
                if cp not in seen_paths:
                    seen_paths.add(cp)
                    checkpoints.append(row)
    if not checkpoints:
        sys.exit(
            f"No checkpoints found in {metadata_files} with merge strategy '{merge}'"
        )
    return checkpoints


def next_checkpoint(metadata_file):
    """Return the first non-completed checkpoint row, or None."""
    with open(metadata_file) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("completed"):
                print(f"Next checkpoint: {row['checkpoint_path']} (threshold {row.get('threshold_value')}, type {row.get('threshold_type')})")
                return row
    return None


def mark_completed(metadata_file, checkpoint_path):
    """Mark a checkpoint as completed in the metadata file."""
    with open(metadata_file) as f:
        lines = f.readlines()
    with open(metadata_file, "w") as f:
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["checkpoint_path"] == checkpoint_path:
                row["completed"] = True
            f.write(json.dumps(row) + "\n")


def print_metadata_paths(paths):
    """Print metadata file paths with a prefix for subprocess IPC."""
    for p in paths:
        print(f"METADATA_FILE:{p}")


def parse_metadata_from_output(output):
    """Extract metadata file paths from subprocess stdout."""
    return [
        line.split(":", 1)[1]
        for line in output.splitlines()
        if line.startswith("METADATA_FILE:")
    ]


def run_dpo(args):
    """Run DPO for the next non-completed checkpoint in the metadata file.

    Processes exactly one checkpoint, marks it completed, then returns.
    Process exit frees all GPU memory.
    """
    metadata_file = args.metadata_file[0]
    checkpoint = next_checkpoint(metadata_file)
    if checkpoint is None:
        print("All checkpoints completed, nothing to do.")
        return

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

    set_chat_template(args.model)
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
    lora_config.use_gradient_checkpointing = True
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

    tags = ["dpo", str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
    if ppl_config is not None:
        tags.append("ppl")

    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="dpo",
        tags=tags,
        config={"stage": "dpo"},
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
    if args.task_name == "gsm8k":
        return [gsm8k_reward_func]
    elif args.task_name == "math500":
        return [math500_reward_func]
    elif args.task_name == "ifeval":
        return [ifeval_reward_func]
    else:
        raise ValueError(f"No reward function for task: {args.task_name}")


def run_grpo(args):
    """Run GRPO for the next non-completed checkpoint in the metadata file.

    Processes exactly one checkpoint, marks it completed, then returns.
    Process exit frees all GPU memory.
    """
    metadata_file = args.metadata_file[0]
    checkpoint = next_checkpoint(metadata_file)
    if checkpoint is None:
        print("All checkpoints completed, nothing to do.")
        return

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

    set_chat_template(args.model)
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
        simple_template=args.grpo_simple_template,
    )
    lora_config = LoraConfig()
    lora_config.use_gradient_checkpointing = True
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

    tags = ["grpo", str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")


    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="grpo",
        tags=tags,
        config={
            "stage": "grpo",
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


def main():
    args = _parse_args()
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
    for metadata_file in all_files:
        if not Path(metadata_file).is_file():
            print(f"Warning: metadata file {metadata_file} does not exist, skipping")
            continue
        while next_checkpoint(metadata_file) is not None:
            pt_cmd = [sys.executable] + base_cmd + [
                pt_flag, "--metadata-file", metadata_file,
            ]
            print(f"[orchestrator] Running {pt_method.upper()}: {' '.join(pt_cmd)}")
            result = subprocess.run(pt_cmd)
            if result.returncode != 0:
                sys.exit(f"{pt_method.upper()} subprocess failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
