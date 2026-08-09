# ABOUTME: CLI parsing, GPU/tier maps, seed init, simplerl resolver, and worker-mode
# ABOUTME: detection. Must not transitively import unsloth/torch/transformers.

import argparse
import os
import random
import sys

import tuning.config
from tuning.training.pipeline.vllm_sidecar import resolve_grpo_server_split


SBATCH_WORKER_SCRIPT_DEFAULT = "tuning/slurm/unified_early_pipeline.sh"
SBATCH_WORKER_SCRIPT_SHORT = "tuning/slurm/unified_early_pipeline_short.sh"


MODEL_TO_GPU_1 = {
    "llama3-1B": 0.85,
    "llama3-1B-instruct": 0.85,
    "llama3-3B": 0.85,
    "llama3-3B-instruct": 0.85,
    "llama3-8B": 0.85,
    "llama3-8B-instruct": 0.85,
    "qwen2-2B": 0.85,
    "qwen2-2B-instruct": 0.85,
    "qwen2-3B": 0.85,
    "qwen2-3B-instruct": 0.85,
    "qwen2-7B": 0.85,
    "qwen2-7B-instruct": 0.85,
    "qwen2-14B": 0.85,
    "qwen2-14B-instruct": 0.85,
    "qwen2-32B": 0.85,
    "qwen2-32B-instruct": 0.85,
    "gemma3-1B": 0.85,
    "gemma3-4B": 0.85,
    "gemma3-12B": 0.85
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
    "qwen2-14B": 0.35,
    "qwen2-14B-instruct": 0.35,
    "qwen2-32B": 0.35,
    "qwen2-32B-instruct": 0.35,
    "gemma3-1B": 0.7,
    "gemma3-4B": 0.62,
    "gemma3-12B": 0.5,
}

MODEL_TO_GPU_3 = {
    "llama3-1B": 0.7,
    "llama3-1B-instruct": 0.7,
    "llama3-3B": 0.45, #0.6
    "llama3-3B-instruct": 0.45,
    "llama3-8B": 0.6, # +0.1 from l8b \delta 6b = 12gb/80 = 0.15
    "llama3-8B-instruct": 0.43,
    "qwen2-2B": 0.45, # 0.45 -> #0.25 after [600, 2048, 8192] -> 5, 10, 31 GB needed
    "qwen2-2B-instruct": 0.45,
    "qwen2-3B": 0.45,
    "qwen2-3B-instruct": 0.45,
    "qwen2-7B": 0.6,
    "qwen2-7B-instruct": 0.45,
    "qwen2-14B": 0.5,
    "qwen2-14B-instruct": 0.3,
    "qwen2-32B": 0.3,
    "qwen2-32B-instruct": 0.3,
    "gemma3-1B": 0.7,
    "gemma3-4B": 0.38,
    "gemma3-12B": 0.5, # from q14
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
    stage.add_argument("--live-dispatch", action="store_true", default=False,
                       help="SFT submits a GRPO post-training worker immediately after "
                            "each pass@k checkpoint is saved, instead of waiting for SFT "
                            "to finish. SFT-to-GRPO only.")
    stage.add_argument("--claim-checkpoint", default=None, metavar="PATH",
                       help="Claim this checkpoint row instead of the next unclaimed "
                            "one. Live-dispatch sets it so each worker trains the "
                            "checkpoint its Slurm allocation was sized for.")
    duration = stage.add_mutually_exclusive_group()
    duration.add_argument("--short", action="store_true", default=False,
                          help="Use the 3-hour sbatch script for better queue times.")
    duration.add_argument("--long", action="store_true", default=False,
                          help="Dispatch jobs to the 24-hour H100 partition.")
    stage.add_argument("--qos", default=None,
                       help="Slurm QOS for every sbatch job the pipeline submits "
                            "(e.g. 'high' for the priority queue).")
    parser.add_argument("--sft-dataset", default="gsm8k", choices=["gsm8k", "tuluif", "tulumix", "ifmix", "openmath", "openmath-lenp95", "openmath-reasoning"])
    parser.add_argument("--dataset", default="gsm8k",
                        choices=["tuluif", "gsm8k", "openmath", "ifrlvr", "dapo",
                                 "mathmix", "simplerl", "simplerl-easy",
                                 "simplerl-medium", "simplerl-hard"])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--sft-data-size", type=int, default=None)
    parser.add_argument("--dpo-data-size", type=int, default=None)
    parser.add_argument("--grpo-data-size", type=int, default=None)
    parser.add_argument("--task-name", default="gsm8k",
                        choices=["ifeval", "gsm8k", "math500", "amc", "ifbench"])
    parser.add_argument("--monitor-evals", nargs="*", default=[],
                        choices=["ifeval", "gsm8k", "math500", "amc", "ifbench"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--sft-max-seq-length", type=int, default=None,
        help="SFT-only context limit; defaults to --max-seq-length.",
    )
    parser.add_argument(
        "--grpo-max-seq-length", type=int, default=None,
        help="GRPO-only context limit; defaults to --max-seq-length.",
    )
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
    parser.add_argument("--sft-full-finetune", action="store_true", default=False,
                        help="Train all weights instead of a LoRA adapter (plain HF/TRL path).")
    parser.add_argument("--sft-mask-prompt-tokens", action=argparse.BooleanOptionalAction, default=True,
                        help="Exclude prompt tokens from the SFT loss for every chat template.")
    parser.add_argument(
        "--sft-optim", type=str, default=None,
        help="SFT optimizer. Defaults to paged_adamw_8bit for full fine-tuning "
             "so pass@k eval can offload optimizer state, otherwise adamw_8bit.",
    )
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-num-epochs", type=int, default=3)
    parser.add_argument("--dpo-eval-steps", type=int, default=256)
    parser.add_argument("--dpo-batch-size", type=int, default=4)
    parser.add_argument("--dpo-grad-accum", type=int, default=4)
    parser.add_argument("--grpo-num-epochs", type=int, default=20)
    parser.add_argument("--grpo-num-gpus", type=int, default=2,
                        help="Total GPUs for GRPO. In colocate mode all are trainer ranks; in server mode "
                             "GPU 0 trains and GPUs 1..N-1 run data-parallel trl vllm-serve workers.")
    parser.add_argument("--grpo-vllm-mode", default="colocate",
                        choices=["colocate", "server"],
                        help="vLLM execution mode for GRPO rollouts. 'server' dedicates GPU 0 to the trainer and "
                             "GPUs 1..N-1 to trl vllm-serve.")
    parser.add_argument("--grpo-vllm-server-host", default="127.0.0.1",
                        help="Host trainer connects to for vLLM server (server mode only).")
    parser.add_argument("--grpo-vllm-server-port", type=int, default=8000,
                        help="Port trainer connects to for vLLM server (server mode only). "
                             "The Python sidecar overrides this when it launches a local server.")
    parser.add_argument("--grpo-vllm-group-port", type=int, default=51216,
                        help="NCCL weight-sync group port (server mode only). "
                             "The Python sidecar overrides this when it launches a local server.")
    parser.add_argument("--grpo-vllm-server-timeout", type=float, default=300.0,
                        help="Seconds the trainer waits for the vLLM server to be reachable (server mode only).")
    parser.add_argument("--grpo-vllm-server-gpu-util", type=float, default=0.9,
                        help="vLLM server gpu_memory_utilization (server mode only).")
    parser.add_argument("--grpo-vllm-sleep-mode",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TRL GRPO vLLM sleep mode in colocate mode. "
                             "Offloads vLLM weights/cache during optimizer steps.")
    parser.add_argument("--grpo-eval-steps", type=int, default=128)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-eval-batch-size", type=int, default=None,
                        help="Per-device eval batch for the GRPO logps forward pass. "
                             "Defaults to the GRPOTrainingConfig value. Lower it for long "
                             "--grpo-max-completion-length runs to avoid eval OOM; it must "
                             "keep (value * num_gpus) divisible by num_generations_eval.")
    parser.add_argument("--grpo-grad-accum", type=int, default=16)
    parser.add_argument("--grpo-num-generations", type=int, default=8)
    parser.add_argument(
        "--grpo-num-generations-eval", type=int, default=None,
        help="Generations per prompt for GRPO trainer evaluation. Defaults to "
             "--grpo-num-generations.",
    )
    parser.add_argument("--grpo-num-iterations", type=int, default=1,
                        help="μ in the GRPO paper: inner optimization passes per rollout batch.")
    parser.add_argument("--grpo-max-completion-length", type=int, default=2048)
    parser.add_argument("--grpo-beta", type=float, default=1e-4)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-warmup-ratio", type=float, default=0.00)
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
    parser.add_argument("--grpo-precision", default="auto",
                        choices=["auto", "fp16", "bf16"],
                        help="GRPO model/vLLM dtype override for fp16/bf16 loading experiments.")
    parser.add_argument("--grpo-use-liger-kernel",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Fused Triton lm_head+GRPO loss (liger-kernel); cuts backward peak memory.")
    parser.add_argument("--grpo-zero-variance-filter",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Drop prompt groups with zero reward variance from GRPO policy loss.")
    parser.add_argument("--grpo-zero-variance-filter-epsilon", type=float, default=0.0,
                        help="Advantage magnitude threshold for GRPO zero-variance filtering.")
    parser.add_argument("--simple-template", action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument("--grpo-gpu-util", type=float, default=None,
                        help="Override MODEL_TO_GPU_3 vLLM GPU utilisation for GRPO (0.0–1.0).")
    parser.add_argument("--lora-rank", type=int, default=128,
                        help="LoRA rank (r) for SFT and GRPO LoRA training.")
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
                        default=[1.1])
    parser.add_argument("--sft-passk-max-checkpoint-gap", type=int, default=None)
    parser.add_argument("--sft-passk-target-data-points", type=int, nargs="+",
                        default=None)
    parser.add_argument("--sft-passk-target-total-minutes", type=float, nargs="+",
                        default=None)
    parser.add_argument("--sft-passk-eval-only-minutes", type=float, nargs="+",
                        default=None)

    parser.add_argument("--sft-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--sft-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--sft-passk-n-samples", type=int, default=1)
    parser.add_argument("--sft-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--sft-passk-max-tokens", type=int, default=4096)
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
    parser.add_argument("--dpo-passk-target-total-minutes", type=float, nargs="+",
                        default=None)
    parser.add_argument("--dpo-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--dpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--dpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--dpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--dpo-passk-max-tokens", type=int, default=4096)
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
    parser.add_argument("--grpo-passk-target-total-minutes", type=float, nargs="+",
                        default=None)
    parser.add_argument("--grpo-passk-early", type=parse_early_tuple, nargs="*",
                        default=[])
    parser.add_argument("--grpo-passk-k-values", type=int, nargs="+", default=[1])
    parser.add_argument("--grpo-passk-n-samples", type=int, default=1)
    parser.add_argument("--grpo-passk-num-prompts", type=int, default=1500)
    parser.add_argument("--grpo-passk-max-tokens", type=int, default=4096)
    parser.add_argument("--grpo-passk-temperature", type=float, default=0.5)
    parser.add_argument("--grpo-passk-strict",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-passk-num-inference-gpus", type=int, default=1)
    parser.add_argument("--grpo-passk-persistent-vllm",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-enable-judge",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Enable DeepSeek judge metrics for GRPO pass@k evals.")
    parser.add_argument("--grpo-judge-model", default="deepseek-v4-flash")
    parser.add_argument("--grpo-judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--grpo-judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--grpo-judge-samples-per-prompt", type=int, default=1,
                        help="Responses judged per prompt; <=0 judges all responses.")
    parser.add_argument("--grpo-judge-concurrency", type=int, default=16)
    parser.add_argument("--grpo-judge-timeout", type=float, default=60.0)
    parser.add_argument("--grpo-judge-max-retries", type=int, default=3)
    parser.add_argument("--grpo-judge-max-tokens", type=int, default=64)

    args = parser.parse_args(argv)

    if args.sft_optim is None:
        args.sft_optim = (
            "paged_adamw_8bit" if args.sft_full_finetune else "adamw_8bit"
        )

    if args.grpo_vllm_mode == "server":
        try:
            resolve_grpo_server_split(args.grpo_num_gpus)
        except ValueError as exc:
            parser.error(str(exc))

    if (args.sft_enable_ppl or args.dpo_enable_ppl) and args.max_seq_length == 1024:
        args.max_seq_length = 4096

    args.sbatch_script = (
        SBATCH_WORKER_SCRIPT_SHORT if args.short else SBATCH_WORKER_SCRIPT_DEFAULT
    )

    return args
