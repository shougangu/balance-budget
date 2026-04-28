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

    if args.run_sft and not args.run_all:
        from tuning.training.pipeline.stages import run_sft
        run_sft(args)
        return

    if args.run_dpo and not args.run_all:
        from tuning.training.pipeline.stages import run_dpo
        run_dpo(args)
        return

    if args.run_grpo and not args.run_all:
        from tuning.training.pipeline.stages import run_grpo
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
