# ABOUTME: Orchestrator: top-level submits SFT+post-training as sbatch jobs and exits.
# ABOUTME: Worker/resume mode (--run-dpo|grpo --run-all) skips SFT and runs one local share.

import os
import subprocess
import sys
from pathlib import Path

from tuning.config import HF_MODEL_MAP
from tuning.training.pipeline.checkpoint_metadata import parse_metadata_from_output
from tuning.training.pipeline.cli import _parse_args
from tuning.training.pipeline.vllm_sidecar import (
    GRPOVLLMServerSidecar,
    resolve_grpo_server_split,
)

# torchrun exports these into worker env; propagating them through sbatch
# leaks the parent's rendezvous config to the child and defeats the per-job
# MASTER_PORT default in unified_early_pipeline.sh.
_TORCHRUN_ENV_VARS = (
    "MASTER_ADDR", "MASTER_PORT",
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "GROUP_WORLD_SIZE",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "WANDB_SERVICE"
)

_LONG_SBATCH_FLAGS = (
    "--partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5",
    "--time=1-00:00:00",
)

_SFT_LONG_SBATCH_FLAGS = (
    "--partition=gpubase_h100_b4,gpubase_h100_b5",
    "--time=3-00:00:00",
)
_SHORT_SBATCH_FLAGS = (
    "--partition=gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5",
    "--time=3:00:00"
)
_TWELVE_HOUR_SBATCH_FLAGS = (
    "--partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5",
    "--time=12:00:00",
)

# Live-dispatch allocation sized to the GRPO share left after an SFT checkpoint
# at this total-minutes target: compute budgets are 240/960/3840 GPU-minutes and
# GRPO accrues 2 GPU-minutes per wall minute, so the worker needs roughly
# (budget - sft_minutes) / 2 wall minutes. Unmapped values use the CLI duration
# flags.
_LIVE_DISPATCH_SBATCH_FLAGS = {
    0: _SFT_LONG_SBATCH_FLAGS,
    60: _SHORT_SBATCH_FLAGS,
    120: _SHORT_SBATCH_FLAGS,
    180: _SHORT_SBATCH_FLAGS,
    240: _TWELVE_HOUR_SBATCH_FLAGS,
    480: _TWELVE_HOUR_SBATCH_FLAGS,
    720: _SHORT_SBATCH_FLAGS,
    960: _SFT_LONG_SBATCH_FLAGS,
    1920: _LONG_SBATCH_FLAGS,
    2880: _TWELVE_HOUR_SBATCH_FLAGS,
}

def _build_base_cmd(argv):
    """Build a stage-neutral subprocess command for worker launches."""
    result = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--parallel", "--metadata-file", "--claim-checkpoint"):
            skip_next = True
            continue
        if tok in (
            "--run-sft", "--run-dpo", "--run-grpo", "--run-all",
            "--dispatch", "--no-dispatch",
        ):
            continue
        result.append(tok)
    return result


def _sbatch_flags_for_args(args):
    """Return scheduler overrides requested by the pipeline CLI."""
    if getattr(args, "long", False) and getattr(args, "run_sft", False):
        return list(_SFT_LONG_SBATCH_FLAGS)
    elif getattr(args, "long", False):
        return list(_LONG_SBATCH_FLAGS)
    elif getattr(args, "short", False):
        return list(_SHORT_SBATCH_FLAGS)
    return []


def _submit_sbatch_worker(sbatch_script, worker_args, sbatch_flags=(), wait=False):
    """Submit an sbatch worker job, return the Slurm job ID as a string.

    sbatch_flags go between 'sbatch' and the script path. When wait=True,
    sbatch blocks until the submitted job terminates and the orchestrator
    exits if the job failed. Exits the orchestrator on sbatch submission
    error or unparseable output.
    """
    flags = list(sbatch_flags)
    if wait:
        flags.append("--wait")
    cmd = ["sbatch", *flags, sbatch_script, *worker_args]
    print(f"[orchestrator] Submitting sbatch worker: {' '.join(cmd)}")
    child_env = {k: v for k, v in os.environ.items() if k not in _TORCHRUN_ENV_VARS}
    result = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    tokens = result.stdout.strip().split()
    if len(tokens) < 4 or tokens[0] != "Submitted":
        sys.exit(f"sbatch submission failed (code {result.returncode}): "
                 f"stdout={result.stdout!r} stderr={result.stderr.strip()}")
    job_id = tokens[-1]
    if result.returncode != 0:
        sys.exit(f"sbatch job {job_id} failed (code {result.returncode}): "
                 f"{result.stderr.strip()}")
    return job_id


def _dispatch_parallel_workers(parallel, base_cmd, pt_flag, metadata_files,
                                sbatch_script, args):
    """Submit sbatch workers for post-training.

    Injects --gres=gpu:N when pt_method=='grpo' and grpo_num_gpus>1.
    """

    sbatch_flags = _sbatch_flags_for_args(args)
    if args.post_training_method == "grpo" and args.grpo_num_gpus > 1:
        sbatch_flags.append(f"--gres=gpu:{args.grpo_num_gpus}")

    worker_argv = list(base_cmd[1:])
    worker_argv += [pt_flag, "--run-all", "--no-dispatch"]
    for mf in metadata_files:
        if Path(mf).is_file():
            worker_argv += ["--metadata-file", mf]
    for i in range(parallel):
        job_id = _submit_sbatch_worker(sbatch_script, worker_argv,
                                        sbatch_flags=sbatch_flags)
        project = getattr(args, "wandb_project", "wandb")
        print(f"[orchestrator] Submitted worker {i+1}/{parallel}: job {job_id}_{project}.out")



def submit_post_training_worker_for_metadata(args, metadata_file,
                                              sft_total_minutes=None,
                                              checkpoint_path=None):
    """Submit one GRPO post-training worker for a freshly saved metadata file.

    Mirrors the worker-arg shape of _dispatch_parallel_workers but submits exactly
    one worker and never aborts the caller: an sbatch failure is logged and
    swallowed so a live-dispatch submission can't kill the running SFT job.

    sft_total_minutes selects an allocation from _LIVE_DISPATCH_SBATCH_FLAGS
    sized to the GRPO share remaining after that checkpoint; unmapped or missing
    values fall back to the CLI duration flags via _sbatch_flags_for_args(args)
    
    checkpoint_path pins the worker
    to that metadata row via --claim-checkpoint so the sized allocation trains
    the checkpoint it was sized for.
    """
    base_cmd = _build_base_cmd(sys.argv)
    worker_argv = list(base_cmd[1:])
    worker_argv += ["--run-grpo", "--run-all", "--no-dispatch"]
    worker_argv += ["--metadata-file", metadata_file]
    if checkpoint_path:
        worker_argv += ["--claim-checkpoint", checkpoint_path]

    args.run_sft = False # ensure the sft + long combination doesn't occur for gpu
    allocation = _LIVE_DISPATCH_SBATCH_FLAGS.get(sft_total_minutes)
    sbatch_flags = (list(allocation) if allocation is not None
                    else _sbatch_flags_for_args(args))
    if args.grpo_num_gpus > 1:
        sbatch_flags.append(f"--gres=gpu:{args.grpo_num_gpus}")

    try:
        job_id = _submit_sbatch_worker(
            args.sbatch_script, worker_argv, sbatch_flags=sbatch_flags,
        )
    except SystemExit as exc:
        print(f"[orchestrator] WARNING: live-dispatch sbatch submission failed "
              f"for {metadata_file}: {exc}")
        return None
    print(f"[orchestrator] Live-dispatched GRPO worker for {metadata_file}: job {job_id}")
    return job_id


def _run_grpo_server_subprocess(base_cmd, pt_flag, metadata_file, args):
    """Start the vLLM sidecar, then run the single-GPU GRPO trainer."""
    split = resolve_grpo_server_split(args.grpo_num_gpus)
    precision_dtype_names = {"fp16": "float16", "bf16": "bfloat16"}
    sidecar_kwargs = dict(
        model_hf=HF_MODEL_MAP[args.model],
        split=split,
        gpu_memory_utilization=args.grpo_vllm_server_gpu_util,
        startup_timeout=args.grpo_vllm_server_timeout,
    )
    vllm_dtype = precision_dtype_names.get(getattr(args, "grpo_precision", "auto"))
    if vllm_dtype is not None:
        sidecar_kwargs["dtype"] = vllm_dtype
    with GRPOVLLMServerSidecar(**sidecar_kwargs) as server:
        trainer_env = {k: v for k, v in os.environ.items() if k not in _TORCHRUN_ENV_VARS}
        trainer_env["CUDA_VISIBLE_DEVICES"] = ",".join(split.trainer_gpus)
        pt_cmd = [
            sys.executable, *base_cmd,
            pt_flag, "--metadata-file", metadata_file,
            "--grpo-vllm-server-host", server.host,
            "--grpo-vllm-server-port", str(server.port),
            "--grpo-vllm-group-port", str(server.group_port),
        ]
        print(
            "[orchestrator] Running GRPO server-mode trainer: "
            f"CUDA_VISIBLE_DEVICES={trainer_env['CUDA_VISIBLE_DEVICES']} "
            f"{' '.join(pt_cmd)}"
        )
        return subprocess.run(pt_cmd, env=trainer_env)


def _run_post_training_subprocess(pt_method, base_cmd, pt_flag, metadata_file, args):
    """Run one claimed post-training subprocess, starting vLLM sidecar if needed."""
    claim = getattr(args, "claim_checkpoint", None)
    if claim:
        base_cmd = base_cmd + ["--claim-checkpoint", claim]
    if pt_method == "grpo" and args.grpo_vllm_mode == "server":
        return _run_grpo_server_subprocess(base_cmd, pt_flag, metadata_file, args)

    if pt_method == "grpo" and args.grpo_num_gpus > 1:
        master_port = 20000 + (os.getpid() % 20000)
        pt_cmd = (
            ["torchrun", f"--nproc_per_node={args.grpo_num_gpus}",
             f"--master-port={master_port}"]
            + base_cmd + [pt_flag, "--metadata-file", metadata_file]
        )
    else:
        pt_cmd = [sys.executable] + base_cmd + [
            pt_flag, "--metadata-file", metadata_file,
        ]
    print(f"[orchestrator] Running {pt_method.upper()}: {' '.join(pt_cmd)}")
    return subprocess.run(pt_cmd)


def main():
    args = _parse_args()
    print(args)

    # run_all if and only if on an orchestrator state
    if not any([args.run_sft, args.run_dpo, args.run_grpo, args.run_all]):
        args.run_all = True
        args.run_sft = True

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

    base_cmd = _build_base_cmd(sys.argv)
    pt_method = args.post_training_method
    pt_flag = f"--run-{pt_method}"

    if args.run_sft and args.run_all:
        sft_argv = list(base_cmd[1:]) + ["--run-sft"]
        job_id = _submit_sbatch_worker(
            args.sbatch_script,
            sft_argv,
            sbatch_flags=_sbatch_flags_for_args(args),
            wait=True,
        )
        sft_output = Path(f"{job_id}_{args.wandb_project}.out").read_text()
        print(sft_output)
        metadata_files = parse_metadata_from_output(sft_output)
        if not metadata_files and not args.metadata_file:
            sys.exit("No metadata files from SFT and no --metadata-file provided")
        all_files = metadata_files + (args.metadata_file or [])
        args.run_sft = False # avoid re-running SFT in workers
    else:
        all_files = args.metadata_file or []
    all_files = list(dict.fromkeys(all_files))
    print(f"Metadata files for post-training: {all_files}")
    
    if args.run_all and args.dispatch:
        _dispatch_parallel_workers(
            parallel=args.parallel,
            base_cmd=base_cmd,
            pt_flag=pt_flag,
            metadata_files=all_files,
            sbatch_script=args.sbatch_script,
            args=args,
        )
        return

    for metadata_file in all_files:
        if not Path(metadata_file).is_file():
            print(f"Warning: metadata file {metadata_file} does not exist, skipping")
            continue
        while True:
            result = _run_post_training_subprocess(
                pt_method, base_cmd, pt_flag, metadata_file, args,
            )
            if result.returncode == 42:
                print(f"[orchestrator] No more checkpoints in {metadata_file}, moving on")
                break
            if result.returncode != 0:
                sys.exit(f"{pt_method.upper()} subprocess failed with return code "
                         f"{result.returncode}")
