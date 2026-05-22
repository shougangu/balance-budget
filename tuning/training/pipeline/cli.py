# ABOUTME: CLI parsing, GPU/tier maps, seed init, simplerl resolver, and worker-mode
# ABOUTME: detection. Must not transitively import unsloth/torch/transformers.

import argparse
import os
import random
import sys

import tuning.config


SBATCH_WORKER_SCRIPT_DEFAULT = "tuning/slurm/unified_early_pipeline.sh"
SBATCH_WORKER_SCRIPT_SHORT = "tuning/slurm/unified_early_pipeline_short.sh"


MODEL_TO_GPU_1 = {
    "llama3-1B": 0.75,
    "llama3-1B-instruct": 0.75,
    "llama3-3B": 0.75,
    "llama3-3B-instruct": 0.75,
    "llama3-8B": 0.6,
    "llama3-8B-instruct": 0.6,
    "qwen2-2B": 0.65,
    "qwen2-2B-instruct": 0.65,
    "qwen2-3B": 0.65,
    "qwen2-3B-instruct": 0.65,
    "qwen2-7B": 0.55,
    "qwen2-7B-instruct": 0.55,
    "gemma3-1B": 0.75,
    "gemma3-4B": 0.65,
    "gemma3-12B": 0.55,
}

MODEL_TO_GPU_2 = {
    "llama3-1B": 0.7,
    "llama3-1B-instruct": 0.7,
    "llama3-3B": 0.62,
    "llama3-3B-instruct": 0.62,
    "llama3-8B": 0.62,
    "llama3-8B-instruct": 0.62,
    "qwen2-2B": 0.65,
    "qwen2-2B-instruct": 0.65,
    "qwen2-3B": 0.65,
    "qwen2-3B-instruct": 0.65,
    "qwen2-7B": 0.5,
    "qwen2-7B-instruct": 0.5,
    "gemma3-1B": 0.7,
    "gemma3-4B": 0.62,
    "gemma3-12B": 0.5,
}

MODEL_TO_GPU_3 = {
    "llama3-1B": 0.7,
    "llama3-1B-instruct": 0.7,
    "llama3-3B": 0.45, #0.6
    "llama3-3B-instruct": 0.45,
    "llama3-8B": 0.43,
    "llama3-8B-instruct": 0.43,
    "qwen2-2B": 0.3, # 0.45 -> #0.25 after [600, 2048, 8192] -> 5, 10, 31 GB needed 
    "qwen2-2B-instruct": 0.45,
    "qwen2-3B": 0.45,
    "qwen2-3B-instruct": 0.45,
    "qwen2-7B": 0.45,
    "qwen2-7B-instruct": 0.45,
    "gemma3-1B": 0.7,
    "gemma3-4B": 0.45,
    "gemma3-12B": 0.4,
}

MODEL_TO_SIMPLERL_TIER = {
    "llama3-1B": "medium",
    "llama3-1B-instruct": "medium",
    "llama3-3B": "medium",
    "llama3-3B-instruct": "medium",
    "llama3-8B": "medium",
    "llama3-8B-instruct": "medium",
    "qwen2-2B":  "medium",
    "qwen2-2B-instruct":  "medium",
    "qwen2-3B":  "medium",
    "qwen2-3B-instruct":  "medium",
    "qwen2-7B":  "medium",
    "qwen2-7B-instruct":  "medium",
    "gemma3-1B": "medium",
    "gemma3-4B": "medium",
    "gemma3-12B": "medium",
}


def init_cuda_env():
    """Restrict training to GPU 0 and save full GPU list for inference workers.

    No-op under torchrun (LOCAL_RANK is set) because each rank's CUDA_VISIBLE_DEVICES
    is already pinned per-rank by the launcher.
    """
    if "LOCAL_RANK" in os.environ:
        return
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES_ALL"] = all_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus.split(",")[0]


def is_worker_mode():
    """True when running as a training worker (needs CUDA), not as orchestrator or in tests."""
    return (
        ("--run-sft" in sys.argv or "--run-dpo" in sys.argv or "--run-grpo" in sys.argv)
        and "--run-all" not in sys.argv
    )


def parse_early_tuple(s):
    """Parse 'patience:min_delta' string into (int, float) tuple."""
    try:
        patience_str, delta_str = s.split(":")
        return (int(patience_str), float(delta_str))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            f"Invalid early tuple {s!r}: expected 'patience:min_delta' (e.g. '2:0.02')"
        )


def effective_eval_seed(seed: int, eval_seed: int | None) -> int:
    """Return eval_seed when set, else the master seed."""
    return eval_seed if eval_seed is not None else seed


def _resolve_simplerl_dataset(args):
    """Rewrite args.dataset='simplerl' to a concrete tier based on model strength."""
    if args.dataset == "simplerl":
        tier = MODEL_TO_SIMPLERL_TIER[args.model]
        print(f"[simplerl] {args.model} -> simplerl-{tier}")
        args.dataset = f"simplerl-{tier}"


def _init_seeds(args):
    """Set global seed state from CLI args. Call once per stage, like set_chat_template().

    Sets tuning.config.DEFAULT_SEED, tuning.config.DEFAULT_EVAL_SEED, and seeds the
    Python random module for data loading.
    """
    from tuning.config import set_seed, set_eval_seed
    set_seed(args.seed)
    set_eval_seed(effective_eval_seed(args.seed, args.eval_seed))
    random.seed(args.seed)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified SFT+DPO pipeline with pass@k and perplexity callbacks."
    )

    parser.add_argument("--model", required=True, choices=list(MODEL_TO_GPU_1),
                        help="Base model name")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")
    parser.add_argument("--tags", nargs="*", default=[], help="W&B run tags")

    stage = parser.add_argument_group("stage control")
    stage.add_argument("--run-sft", action="store_true", default=False)
    stage.add_argument("--run-dpo", action="store_true", default=False)
    stage.add_argument("--run-grpo", action="store_true", default=False)
    stage.add_argument("--run-all", action="store_true", default=False)
    stage.add_argument("--post-training-method", default="dpo",
                       choices=["dpo", "grpo", "kto"])
    stage.add_argument("--parallel", type=int, default=1,
                       help="Number of concurrent post-training workers.")
    stage.add_argument("--dispatch", action=argparse.BooleanOptionalAction,
                       default=True)
    stage.add_argument("--short", action="store_true", default=False,
                       help="Use the shorter sbatch script for better queue times.")
    parser.add_argument("--sft-dataset", default="gsm8k", choices=["gsm8k", "tuluif", "openmath", "openmath-lenp95", "openmath-reasoning"])
    parser.add_argument("--dataset", default="gsm8k",
                        choices=["tuluif", "gsm8k", "openmath", "ifrlvr",
                                 "simplerl", "simplerl-easy", "simplerl-medium",
                                 "simplerl-hard"])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--sft-data-size", type=int, default=None)
    parser.add_argument("--dpo-data-size", type=int, default=None)
    parser.add_argument("--grpo-data-size", type=int, default=None)
    parser.add_argument("--task-name", default="gsm8k",
                        choices=["ifeval", "gsm8k", "math500", "ifbench"])
    parser.add_argument("--monitor-evals", nargs="*", default=[],
                        choices=["ifeval", "gsm8k", "math500", "ifbench"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--metadata-merge", choices=["union", "passk", "ppl"],
                        default="union")
    parser.add_argument("--metadata-file", action="append", dest="metadata_file",
                        metavar="FILE")

    parser.add_argument("--sft-resume", type=str, default="",
                        help="W&B run ID to resume SFT from. Empty = start fresh.")
    parser.add_argument("--sft-learning-rate", type=float, default=5e-5)
    parser.add_argument("--sft-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--sft-lr-scheduler-type", type=str, default="constant")
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
    parser.add_argument("--grpo-num-gpus", type=int, default=2,
                        help="Number of GPUs for GRPO DDP training. >1 launches GRPO via torchrun.")
    parser.add_argument("--grpo-eval-steps", type=int, default=64)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-grad-accum", type=int, default=32)
    parser.add_argument("--grpo-num-generations", type=int, default=8)
    parser.add_argument("--grpo-num-iterations", type=int, default=1,
                        help="μ in the GRPO paper: inner optimization passes per rollout batch.")
    parser.add_argument("--grpo-max-completion-length", type=int, default=2048)
    parser.add_argument("--grpo-beta", type=float, default=0.0)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grpo-lr-scheduler-type", type=str, default="constant")
    parser.add_argument("--grpo-loss-type", default="dapo",
                        choices=["grpo", "dr_grpo", "dapo", "bnpo", "cispo"])
    parser.add_argument("--grpo-scale-rewards", default="group",
                        choices=["group", "batch", "false"])
    parser.add_argument("--grpo-vllm-importance-sampling",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable vLLM IS correction (only meaningful with --grpo-num-iterations > 1).")
    parser.add_argument("--grpo-upcast-lm-head-fp32",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="MiniMax/ScaleRL fp32 lm_head on trainer + vLLM for GRPO stability.")
    parser.add_argument("--grpo-use-liger-kernel",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Fused Triton lm_head+GRPO loss (liger-kernel); cuts backward peak memory.")
    parser.add_argument("--simple-template", action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument("--grpo-gpu-util", type=float, default=None,
                        help="Override MODEL_TO_GPU_3 vLLM GPU utilisation for GRPO (0.0–1.0).")
    parser.add_argument("--grpo-lora-target-modules", type=str, nargs="+", default=None)
    parser.add_argument("--grpo-lora-layers-fraction", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Gradient checkpointing for SFT and GRPO LoRA training. "
                             "--no-gradient-checkpointing disables it (faster step, much higher activation memory).")

    parser.add_argument("--sft-enable-passk", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--sft-enable-ppl", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--dpo-enable-passk", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dpo-enable-ppl", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--grpo-enable-passk", action=argparse.BooleanOptionalAction,
                        default=True)

    parser.add_argument("--sft-passk-targets", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--sft-passk-max-checkpoint-gap", type=int, default=None)
    parser.add_argument("--sft-passk-target-data-points", type=int, nargs="+",
                        default=None)
    
    parser.add_argument("--sft-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--sft-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--sft-passk-n-samples", type=int, default=1)
    parser.add_argument("--sft-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--sft-passk-temperature", type=float, default=0.5)
    parser.add_argument("--sft-passk-strict",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sft-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--sft-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sft-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--sft-ppl-num-samples", type=int, default=541)
    parser.add_argument("--sft-ppl-early", type=parse_early_tuple, nargs="*", default=[])

    parser.add_argument("--dpo-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--dpo-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--dpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--dpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--dpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--dpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--dpo-passk-strict",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--dpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--dpo-ppl-thresholds", type=float, nargs="+", default=[1.0])
    parser.add_argument("--dpo-ppl-num-samples", type=int, default=541)
    parser.add_argument("--dpo-ppl-early", type=parse_early_tuple, nargs="*", default=[])

    parser.add_argument("--grpo-passk-targets", type=float, nargs="+", default=[1.2])
    parser.add_argument("--grpo-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--grpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--grpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--grpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--grpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--grpo-passk-strict",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--grpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args(argv)

    if (args.sft_enable_ppl or args.dpo_enable_ppl) and args.max_seq_length == 1024:
        args.max_seq_length = 4096

    args.sbatch_script = (
        SBATCH_WORKER_SCRIPT_SHORT if args.short else SBATCH_WORKER_SCRIPT_DEFAULT
    )

    return args
