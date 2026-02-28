# ABOUTME: CLI-driven unified SFT+DPO pipeline with optional pass@k and perplexity callbacks.
# ABOUTME: Supports SFT-only, DPO-only, and full SFT→DPO runs from a single command.

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
    return ("--run-sft" in sys.argv or "--run-dpo" in sys.argv) and "--run-all" not in sys.argv


if _is_worker_mode():
    _init_cuda_env()
    import unsloth  # noqa: F401 - must be imported before trl/transformers/peft


MODEL_TO_GPU_1 = {
    "llama3-1B": 0.5,
    "llama3-3B": 0.5,    # (0.65 gives 76% peak with non-persistent vLLM with one 97% spike?)
    "llama3-8B": 0.45,
    "qwen2-3B": 0.5,     # (0.65 gives 76% peak with non-persistence but one 91% spike?)
    "qwen2-7B": 0.5,
}
MODEL_TO_GPU_2 = {
    "llama3-1B": 0.45,
    "llama3-3B": 0.45,  # can reach 
    "llama3-8B": 0.4,
    "qwen2-3B": 0.45,
    "qwen2-7B": 0.45,
}


MODEL_TO_GPU_1 = {
"llama3-1B": 0.75,
"llama3-3B": 0.65, # (0.65 gives 76% peak with non-persistent vLLM with one 97% spike?)
"llama3-8B": 0.68,
"qwen2-3B": 0.65, # (0.65 gives 76% peak with non-persistence but one 91% spike?)
"qwen2-7B": 0.55,
}
# MODEL_TO_GPU_2 = {
# "llama3-1B": 0.7,
# "llama3-3B": 0.62, # can reach
# "llama3-8B": 0.45,
# "qwen2-3B": 0.62,
# "qwen2-7B": 0.45,
# }


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
    stage.add_argument("--run-all", action="store_true", default=False,
                       help="Explicitly run both stages (default when none specified)")

    # Core
    parser.add_argument("--dataset", default="tuluif")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--task-name", default="ifeval")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-merge", choices=["union", "passk", "ppl"], default="union",
                        help="Which checkpoint types to use for DPO")
    parser.add_argument("--metadata-file", action="append", dest="metadata_file",
                        metavar="FILE",
                        help="Metadata JSONL file from a previous SFT run (repeatable)")

    # Training args
    parser.add_argument("--sft-eval-steps", type=int, default=64)
    parser.add_argument("--sft-batch-size", type=int, default=16)
    parser.add_argument("--sft-grad-accum", type=int, default=1)
    parser.add_argument("--dpo-eval-steps", type=int, default=64)
    parser.add_argument("--dpo-batch-size", type=int, default=4)
    parser.add_argument("--dpo-grad-accum", type=int, default=4)

    # Callback toggles
    parser.add_argument("--sft-enable-passk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft-enable-ppl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dpo-enable-passk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpo-enable-ppl", action=argparse.BooleanOptionalAction, default=False)

    # SFT pass@k
    parser.add_argument("--sft-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--sft-passk-early", type=parse_early_tuple, nargs="*",
                        default=[(1, 0.02), (2, 0.02), (3, 0.02), (4, 0.02), (5, 0.02)])
    parser.add_argument("--sft-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--sft-passk-n-samples", type=int, default=1)
    parser.add_argument("--sft-passk-num-prompts", type=int, default=541)
    parser.add_argument("--sft-passk-temperature", type=float, default=0.5)
    parser.add_argument("--sft-passk-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--sft-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    # SFT perplexity
    parser.add_argument("--sft-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--sft-ppl-num-samples", type=int, default=541)
    parser.add_argument("--sft-ppl-early", type=parse_early_tuple, nargs="*",
                        default=[(1, 0.02), (2, 0.02), (3, 0.02), (4, 0.02), (5, 0.02)])

    # DPO pass@k
    parser.add_argument("--dpo-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--dpo-passk-early", type=parse_early_tuple, nargs="*", default=[])
    parser.add_argument("--dpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--dpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--dpo-passk-num-prompts", type=int, default=541)
    parser.add_argument("--dpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--dpo-passk-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--dpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    # DPO perplexity
    parser.add_argument("--dpo-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dpo-ppl-num-samples", type=int, default=541)
    parser.add_argument("--dpo-ppl-early", type=parse_early_tuple, nargs="*", default=[])


    return parser.parse_args(argv)


def _sft_passk_config(args, gpu_util):
    """Return PassAtKConfig for SFT, or None if disabled."""
    if not args.sft_enable_passk:
        return None
    from tuning.training.config_training import PassAtKConfig
    return PassAtKConfig(
        target_pass_at_k=args.sft_passk_targets,
        early_tuples=args.sft_passk_early or None,
        k_values=args.sft_passk_k_values,
        n_samples=args.sft_passk_n_samples,
        num_prompts=args.sft_passk_num_prompts,
        temperature=args.sft_passk_temperature,
        strict=args.sft_passk_strict,
        enabled=True,
        num_inference_gpus=args.sft_passk_num_inference_gpus,
        use_persistent_vllm=args.sft_passk_persistent_vllm,
        vllm_gpu_memory_utilization=gpu_util,
    )


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


def _dpo_passk_config(args, gpu_util):
    """Return PassAtKConfig for DPO, or None if disabled."""
    if not args.dpo_enable_passk:
        return None
    from tuning.training.config_training import PassAtKConfig
    return PassAtKConfig(
        target_pass_at_k=args.dpo_passk_targets,
        early_tuples=args.dpo_passk_early or None,
        k_values=args.dpo_passk_k_values,
        n_samples=args.dpo_passk_n_samples,
        num_prompts=args.dpo_passk_num_prompts,
        temperature=args.dpo_passk_temperature,
        strict=args.dpo_passk_strict,
        enabled=True,
        num_inference_gpus=args.dpo_passk_num_inference_gpus,
        use_persistent_vllm=args.dpo_passk_persistent_vllm,
        vllm_gpu_memory_utilization=gpu_util,
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


def _sft_tags(passk_config, ppl_config):
    """Build W&B tags for an SFT run."""
    from tuning.training.wandb_utils import get_early_pairs, early_pair_tag
    tags = ["sft"]
    if passk_config is not None:
        tags.append(f"p{passk_config.k_values[0]}")
        tags.append(early_pair_tag(get_early_pairs(passk_config)))
    if ppl_config is not None:
        tags.append("ppl")
        tags.append(early_pair_tag(get_early_pairs(ppl_config)))
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
    training_args.eval_steps = args.sft_eval_steps
    training_args.per_device_train_batch_size = args.sft_batch_size
    training_args.gradient_accumulation_steps = args.sft_grad_accum

    passk_config = _sft_passk_config(args, gpu_util)
    ppl_config = _sft_ppl_config(args)
    tags = _sft_tags(passk_config, ppl_config)

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
        )

    metadata_paths = [
        cb.metadata_path
        for cb in callbacks
        if getattr(cb, "metadata_path", None)
    ]

    del model, tokenizer, trainer, callbacks
    cleanup_gpu()
    print(subprocess.check_output("nvidia-smi").decode())
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
                return row
    return None


def mark_completed(metadata_file, checkpoint_path):
    """Mark a checkpoint as completed in the metadata file."""
    with open(metadata_file) as f:
        lines = f.readlines()
    with open(metadata_file, "w") as f:
        for line in lines:
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


def run_dpo(args, checkpoints):
    """Run DPO stage for each checkpoint in the list."""
    import torch
    torch.cuda.memory._record_memory_history(max_entries=100000)
    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
        LoraConfig, DPOTrainingConfig,
    )
    from tuning.training.dpo_training import train_model_dpo
    from tuning.utils.gpu import cleanup_gpu

    set_chat_template(args.model)
    gpu_util = MODEL_TO_GPU_2[args.model]

    for checkpoint in checkpoints:
        remaining = args.train_size - checkpoint["data_points_seen"]
        if remaining <= 0:
            print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
            continue

        model_name = Path(checkpoint["checkpoint_path"]).name

        dataset_config = DatasetConfig(
            dataset=args.dataset,
            dataset_type="pt",
            train_size=remaining,
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
        training_args.eval_steps = args.dpo_eval_steps
        training_args.per_device_train_batch_size = args.dpo_batch_size
        training_args.gradient_accumulation_steps = args.dpo_grad_accum

        passk_config = _dpo_passk_config(args, gpu_util)
        ppl_config = _dpo_ppl_config(args)

        tags = ["dpo", str(checkpoint["threshold_value"])]
        if passk_config is not None:
            tags.append(f"p{passk_config.k_values[0]}")
        if ppl_config is not None:
            tags.append("ppl")

        with wandb.init(
            name=run_config.model_name,
            project=args.wandb_project,
            job_type="dpo",
            tags=tags,
            config={"stage": "dpo"},
            reinit=True,
        ):
            model, tokenizer, trainer, _ = train_model_dpo(
                run_config=run_config,
                lora_config=lora_config,
                model_load_config=model_load_config,
                training_args=training_args,
                passk_config=passk_config,
                perplexity_config=ppl_config,
            )

        wandb.finish(quiet=True)
        del trainer.ref_model
        del trainer.model
        trainer.accelerator.free_memory()
        del model, tokenizer, trainer, _
        cleanup_gpu()
        print(subprocess.check_output("nvidia-smi").decode())
        import torch
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[Memory] allocated={alloc:.1f}GiB reserved={reserved:.1f}GiB")
    torch.cuda.memory._dump_snapshot("profile.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)




def main():
    args = _parse_args()
    print(args)

    if not any([args.run_sft, args.run_dpo, args.run_all]):
        args.run_sft = True
        args.run_dpo = True

    metadata_files = []

    if args.run_sft or args.run_all:
        metadata_files = run_sft(args)

    if args.run_dpo or args.run_all:
        if not metadata_files and not args.metadata_file:
            sys.exit("DPO stage requires --metadata-file (or run SFT first)")
        files = metadata_files + (args.metadata_file or [])
        checkpoints = load_checkpoints(files, args.metadata_merge)
        run_dpo(args, checkpoints)


if __name__ == "__main__":
    main()
