# ABOUTME: Evaluates one banked budget-mark checkpoint offline on the math suite at long
# ABOUTME: generation caps, logging scores to W&B grouped under the parent training run.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BENCHMARKS = "olympiadbench,amc,aime24,aime25,hmmt_feb25,gsm8k,math500"
TEMPLATES = ("auto", "simple", "repo", "native")


def checkpoint_total_minutes(checkpoint: str) -> float:
    """Budget position of the checkpoint, read from its saved trainer state."""
    from tuning.training.callback_utils import load_total_seconds_from_checkpoint
    return load_total_seconds_from_checkpoint(checkpoint, warn=False) / 60.0


def resolve_total_minutes(args) -> float:
    """The metadata row's value when the submitter passed it (verl's HF export
    saves no trainer state), else the clock saved inside the checkpoint."""
    if args.total_minutes is not None:
        return float(args.total_minutes)
    return checkpoint_total_minutes(args.checkpoint)


def resolve_template(checkpoint: str, template: str) -> str:
    """Map --template auto onto the template the checkpoint was trained with.

    The saved tokenizer carries the training template: ours ("simple") or the
    model family's ("repo", what --no-simple-template trains with).
    """
    if template != "auto":
        return template
    from tuning.utils.utils import on_disk_template_is_simple
    is_simple = on_disk_template_is_simple(checkpoint)
    if is_simple is None:
        raise ValueError(
            f"{checkpoint} saves no chat_template; pass --template explicitly")
    return "simple" if is_simple else "repo"


def wandb_group_from_checkpoint(checkpoint: str) -> str:
    """The parent run id save_sweetspot_checkpoint suffixes onto the dir name."""
    name = os.path.basename(os.path.normpath(checkpoint))
    return name.rsplit("_", 1)[-1] if "_" in name else ""


def flatten_report(report: dict) -> dict:
    """One flat eval/<benchmark>/<metric> mapping for a single W&B log call."""
    return {
        f"eval/{benchmark}/{metric}": value
        for benchmark, scores in report["benchmarks"].items()
        for metric, value in scores.items()
    }


def build_calibration_argv(args, template: str) -> list[str]:
    """Arguments for external_eval_calibration.run, which owns the vLLM loop."""
    argv = [
        "--model", args.checkpoint,
        "--model-family", args.model_family,
        "--benchmarks", args.benchmarks,
        "--template", template,
        "--prompt-style", "ours",
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--n-samples", str(args.n_samples),
        "--amc-n-samples", str(args.amc_n_samples),
        "--aime-n-samples", str(args.aime_n_samples),
        "--k-values", *[str(k) for k in args.k_values],
        "--out", args.out,
    ]
    if args.num_prompts is not None:
        argv += ["--num-prompts", str(args.num_prompts)]
    if args.save_generations:
        argv.append("--save-generations")
    return argv


def log_to_wandb(args, report: dict, total_minutes: float) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"eval_{os.path.basename(os.path.normpath(args.checkpoint))}",
        job_type="offline-eval",
        tags=["offline-eval"],
        config={
            "checkpoint_path": args.checkpoint,
            "sft_wandb_run_id": args.wandb_group,
            "total_minutes": total_minutes,
            "model_family": args.model_family,
            "max_tokens": args.max_tokens,
            "benchmarks": args.benchmarks,
        },
    )
    # Late logging at the parent's budget position: eval curves share the
    # training runs' x-axis instead of this run's own step counter.
    run.define_metric("train/total_minutes")
    run.define_metric("eval/*", step_metric="train/total_minutes")
    run.log({**flatten_report(report), "train/total_minutes": total_minutes})
    run.finish()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline eval of one banked checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="HF-format checkpoint dir")
    parser.add_argument("--model-family", required=True, help="e.g. qwen3-8B")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", default=None,
                        help="Parent training run id; default parsed from the checkpoint name")
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--template", default="auto", choices=TEMPLATES,
                        help="Chat template for prompts; auto reads the checkpoint's saved one")
    parser.add_argument("--total-minutes", type=float, default=None,
                        help="Budget position to log at; default reads the checkpoint's "
                             "trainer state (absent on verl-exported RL marks)")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=36864)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples per problem on the large sets (OlympiadBench, "
                             "MATH-500, GSM8K), which read cleanly without averaging.")
    parser.add_argument("--amc-n-samples", type=int, default=8)
    parser.add_argument("--aime-n-samples", type=int, default=16,
                        help="Samples for the 30-problem sets (AIME 24/25/26, HMMT).")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--save-generations", action=argparse.BooleanOptionalAction,
                        default=True, help="Keep generations.jsonl beside the report for regrading")
    parser.add_argument("--out", default=None,
                        help="Report path; default <checkpoint>/offline_eval.json")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(args.checkpoint, "offline_eval.json")
    if args.wandb_group is None:
        args.wandb_group = wandb_group_from_checkpoint(args.checkpoint)
    return args


def main(argv=None):
    args = parse_args(argv)
    import external_eval_calibration as calibration

    total_minutes = resolve_total_minutes(args)
    template = resolve_template(args.checkpoint, args.template)
    print(f"[offline-eval] checkpoint={args.checkpoint} total_minutes={total_minutes:.1f} "
          f"template={template} group={args.wandb_group}")
    report = calibration.run(calibration.parse_args(build_calibration_argv(args, template)))
    report["total_minutes"] = total_minutes
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    if not args.no_wandb:
        log_to_wandb(args, report, total_minutes)
    return report


if __name__ == "__main__":
    main()
