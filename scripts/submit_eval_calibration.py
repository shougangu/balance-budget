# ABOUTME: Submits the eval-harness calibration sweep: external post-trained models
# ABOUTME: evaluated under our protocol and under the protocol their papers report.

"""Submit one Slurm job per (model, arm) calibration cell.

The arms decompose the gap between an externally reported number and what our
harness measures on the very same weights:

  ours_simple   our exact training-time protocol (SIMPLE_TEMPLATE, our system
                message + "Problem: ...\\nAnswer:", temperature 0.5)
  ours_native   same prompts and decoding, but the model's own chat template
  theirs_sampled the reference boxed instruction on the native template, still
                sampled at temperature 0.5
  theirs_greedy the reference protocol: boxed instruction, native template, greedy

  ours_greedy   our prompts and template, greedy, one sample
  ours_maj256   our prompts and template, 256 samples on MATH-500 (64 on AMC /
                GSM8K) for maj@{4,16,64,256}
  theirs_maj256 the reference protocol sampled 256 times for the same maj@k

  if_ours       our instruction-following protocol (family template + our IF
                system message, temperature 0.5)
  if_native     bare user turn on the native template

Our own checkpoints (l8b-64h-*) are LoRA adapter dirs; the calibration script
serves them on their base and merges the SFT parent first for GRPO adapters.

Usage:
    python scripts/submit_eval_calibration.py --dry-run
    python scripts/submit_eval_calibration.py --models openmath2 llama31-8b-instruct
"""

import argparse
import subprocess

SBATCH_SCRIPT = "tuning/slurm/eval_calibration.sh"
OUT_DIR = "outputs/eval_calibration"
# Merged base+SFT-parent weights for GRPO adapters (~16 GB each) go to scratch,
# not the job's RAM-backed $SLURM_TMPDIR; the path exists on every cluster.
MERGE_ROOT = "/scratch/shougan/balance-budget/tmp/eval_calibration_merge"

MODELS = {
    "openmath2": ("nvidia/OpenMath2-Llama3.1-8B", "llama3-8B"),
    "llama31-8b-instruct": ("unsloth/Meta-Llama-3.1-8B-Instruct", "llama3-8B"),
    "llama32-3b-instruct": ("unsloth/Llama-3.2-3B-Instruct", "llama3-3B"),
    "tulu3-8b": ("allenai/Llama-3.1-Tulu-3-8B", "llama3-8B"),
    "gemma3-12b-it": ("unsloth/gemma-3-12b-it", "gemma3-12B"),
    "gemma3-4b-it": ("unsloth/gemma-3-4b-it", "gemma3-4B"),
    # oz7gxjiw lineage, 64 GPU-h: all SFT, and 25% SFT + 75% GRPO (71xr216b).
    "l8b-64h-100": ("tuning/models/llama3-8B_math500-p@1-3840m_sft-5931696_oz7gxjiw",
                    "llama3-8B"),
    "l8b-64h-25": ("tuning/models/llama3-8B_math500-p@1-3840m_sft-205568_71xr216b",
                   "llama3-8B"),
}

GREEDY = ["--temperature", "0.0", "--top-p", "1.0", "--top-k", "-1",
          "--n-samples", "1", "--amc-n-samples", "1", "--k-values", "1"]
MAJ256 = ["--temperature", "0.5", "--n-samples", "256", "--amc-n-samples", "64",
          "--gsm8k-n-samples", "64", "--k-values", "1", "4", "16", "64", "256"]
SMOKE = ["--num-prompts", "20", "--n-samples", "8", "--amc-n-samples", "8",
         "--gsm8k-n-samples", "8", "--k-values", "1", "4", "8"]

MATH_ARMS = {
    "ours_simple": ["--template", "simple", "--prompt-style", "ours",
                    "--temperature", "0.5"],
    "ours_native": ["--template", "native", "--prompt-style", "ours",
                    "--temperature", "0.5"],
    "theirs_sampled": ["--template", "native", "--prompt-style", "boxed",
                       "--temperature", "0.5"],
    "theirs_greedy": ["--template", "native", "--prompt-style", "boxed", *GREEDY],
    "ours_greedy": ["--template", "simple", "--prompt-style", "ours", *GREEDY],
    "ours_maj256": ["--template", "simple", "--prompt-style", "ours", *MAJ256],
    "theirs_maj256": ["--template", "native", "--prompt-style", "boxed", *MAJ256],
}

IF_ARMS = {
    "if_ours": ["--template", "repo", "--prompt-style", "ours",
                "--temperature", "0.5", "--n-samples", "1", "--k-values", "1"],
    "if_native": ["--template", "native", "--prompt-style", "plain",
                  "--temperature", "0.5", "--n-samples", "1", "--k-values", "1"],
}

MATH_BENCHMARKS = "math500,gsm8k,amc"
IF_BENCHMARKS = "ifeval,ifbench"


def cells(model_keys, suites, arms=None, smoke=False):
    """Yield (key, model, family, arm, benchmarks, flags) for every requested cell.

    smoke appends the SMOKE flags, so a cell runs 20 prompts at n=8 and the json
    can be read before the full sweep is launched.
    """
    for key in model_keys:
        model, family = MODELS[key]
        grid = []
        if "math" in suites:
            grid += [(arm, MATH_BENCHMARKS, flags) for arm, flags in MATH_ARMS.items()]
        if "if" in suites:
            grid += [(arm, IF_BENCHMARKS, flags) for arm, flags in IF_ARMS.items()]
        for arm, benchmarks, flags in grid:
            if arms and arm not in arms:
                continue
            yield key, model, family, arm, benchmarks, flags + SMOKE if smoke else flags


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        choices=list(MODELS))
    parser.add_argument("--suites", nargs="+", default=["math", "if"],
                        choices=["math", "if"])
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Restrict to these arm names.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--smoke", action="store_true",
                        help="20 prompts at n=8 per cell; output name gets a _smoke suffix.")
    parser.add_argument("--sbatch-args", action="append", default=[],
                        help="Extra sbatch flags, e.g. cluster partition/gres/time overrides.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    submitted = []
    for key, model, family, arm, benchmarks, flags in cells(
            args.models, args.suites, args.arms, args.smoke):
        name = f"{key}__{arm}" + ("_smoke" if args.smoke else "")
        cmd = [
            "sbatch", f"--export=ALL,ARM_NAME={name}", *args.sbatch_args, SBATCH_SCRIPT,
            "--model", model,
            "--model-family", family,
            "--benchmarks", benchmarks,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--save-generations",
            "--out", f"{OUT_DIR}/{name}.json",
            *flags,
        ]
        if model.startswith("tuning/models/"):
            cmd += ["--merge-root", MERGE_ROOT]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"{job_id}  {name}")
        submitted.append((job_id, name))

    if submitted:
        print(f"\nSubmitted {len(submitted)} calibration jobs.")


if __name__ == "__main__":
    main()
