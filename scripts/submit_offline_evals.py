# ABOUTME: Watches checkpoint metadata files and sbatches one offline eval per banked row.
# ABOUTME: Rows are flagged eval_submitted so every checkpoint is evaluated exactly once.

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tuning.training.pipeline.checkpoint_metadata import mark_eval_submitted  # noqa: E402

OFFLINE_EVAL_SBATCH = "tuning/slurm/offline_eval.sh"


def pending_rows(metadata_file: str) -> list[dict]:
    """Rows whose checkpoint exists on disk and has no eval submitted yet."""
    rows = []
    with open(metadata_file) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("eval_submitted"):
                continue
            if not os.path.isdir(row.get("checkpoint_path", "")):
                continue
            rows.append(row)
    return rows


def submit_eval(metadata_file: str, row: dict, args) -> str:
    """sbatch one offline eval for the row, then flag it eval_submitted."""
    command = ["sbatch", f"--gres={args.gres}", f"--time={args.time}"]
    if args.partition:
        command.append(f"--partition={args.partition}")
    command += [
        args.sbatch_script,
        "--checkpoint", row["checkpoint_path"],
        "--model-family", args.model_family,
        "--wandb-project", args.wandb_project,
    ]
    if row.get("sft_wandb_run_id"):
        command += ["--wandb-group", row["sft_wandb_run_id"]]
    if row.get("total_minutes") is not None:
        command += ["--total-minutes", str(row["total_minutes"])]
    if args.tensor_parallel_size:
        command += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    command += args.extra_eval_args
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    mark_eval_submitted(metadata_file, row["checkpoint_path"])
    print(f"[submit-offline-evals] {row['checkpoint_path']} -> {output.strip()}")
    return output


def scan_once(args) -> int:
    submitted = 0
    for pattern in args.metadata_files:
        for metadata_file in sorted(glob.glob(pattern)):
            for row in pending_rows(metadata_file):
                submit_eval(metadata_file, row, args)
                submitted += 1
    return submitted


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit offline evals for newly banked checkpoints.")
    parser.add_argument("--metadata-files", nargs="+", required=True,
                        help="Metadata JSONL paths or globs to watch")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--gres", default="gpu:h100:1")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--partition", default=None,
                        help="Override the sbatch script's partition list")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--sbatch-script", default=OFFLINE_EVAL_SBATCH)
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between scans; default is a single pass")
    parser.add_argument("--extra-eval-args", nargs=argparse.REMAINDER, default=[],
                        help="Everything after this flag is passed to offline_checkpoint_eval")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    while True:
        submitted = scan_once(args)
        if args.interval is None:
            return submitted
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
