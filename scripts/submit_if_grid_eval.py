# ABOUTME: Re-evaluates the fifteen instruction-following budget-grid checkpoints at
# ABOUTME: IFEval n=4 and IFBench n=8, one Slurm job per cell.

"""Score the [1]if-g12b budget grid with the repeated sampling the grid needs.

Every cell of the published grid was evaluated once per prompt, which carries a
1.3pt (IFEval) / 2.5pt (IFBench) standard error -- larger than the allocation
differences the grid is meant to resolve. Training now scores both benchmarks at
the fixed (n_samples, k_values) in FIXED_EVAL_SAMPLING, but the checkpoints
already banked were not, so each mark checkpoint is served again here under the
protocol its run trained and evaluated with: the family template, our
instruction-following system message, temperature 0.5, IFEval loose, IFBench
strict.

Each cell writes outputs/if_grid_n48/<budget>h_<fraction>.json, which
research/scripts/refresh_budget_grid.py folds back into the grid.

Usage:
    python scripts/submit_if_grid_eval.py --dry-run
    python scripts/submit_if_grid_eval.py --cells 4h_100 64h_25
"""

import argparse
import subprocess

SBATCH_SCRIPT = "tuning/slurm/if_grid_eval.sh"
OUT_DIR = "outputs/if_grid_n48"
MODEL_FAMILY = "gemma3-12B"
# Base+SFT-parent merged weights for a GRPO adapter; the job deletes its own.
MERGE_ROOT = "/scratch/shougan/balance-budget/tmp/if_grid_merge"

MODELS_DIR = "tuning/models"
CELLS = {
    (4, 0): f"{MODELS_DIR}/gemma3-12B_p@1-240m_sft-2432_cp50g9mh",
    (4, 25): f"{MODELS_DIR}/gemma3-12B_p@1-240m_sft-1920_3ukmz2v6",
    (4, 50): f"{MODELS_DIR}/gemma3-12B_p@1-240m_sft-1280_8ossnghk",
    (4, 75): f"{MODELS_DIR}/gemma3-12B_p@1-240m_sft-640_ux8rhlvp",
    (4, 100): f"{MODELS_DIR}/gemma3-12B_p@1-240m_sft-62608_xjwncuw7",
    (16, 0): f"{MODELS_DIR}/gemma3-12B_p@1-960m_sft-9344_cp50g9mh",
    (16, 25): f"{MODELS_DIR}/gemma3-12B_p@1-960m_sft-7552_8wy8tyib",
    (16, 50): f"{MODELS_DIR}/gemma3-12B_p@1-960m_sft-4992_kto0tywb",
    (16, 75): f"{MODELS_DIR}/gemma3-12B_p@1-960m_sft-2304_39cpniza",
    (16, 100): f"{MODELS_DIR}/gemma3-12B_p@1-960m_sft-252080_xjwncuw7",
    (64, 0): f"{MODELS_DIR}/gemma3-12B_p@1-3840m_sft-37760_cp50g9mh",
    (64, 25): f"{MODELS_DIR}/gemma3-12B_p@1-3840m_sft-34176_40ybac08",
    (64, 50): f"{MODELS_DIR}/gemma3-12B_p@1-3840m_sft-29056_0n0us5j6",
    (64, 75): f"{MODELS_DIR}/gemma3-12B_p@1-3840m_sft-12544_xohlmsqg",
    (64, 100): f"{MODELS_DIR}/gemma3-12B_p@1-3840m_sft-1003104_xjwncuw7",
}

# The protocol the [1]if-g12b runs trained and evaluated under, with the sample
# counts FIXED_EVAL_SAMPLING now pins: IFEval n=4, IFBench n=8.
PROTOCOL = [
    "--benchmarks", "ifeval,ifbench",
    "--template", "repo",
    "--prompt-style", "ours",
    "--temperature", "0.5",
    "--n-samples", "4",
    "--ifbench-n-samples", "8",
    "--k-values", "1", "2", "4", "8",
]


def cell_name(budget: int, fraction: int) -> str:
    return f"{budget}h_{fraction}"


def cell_flags(budget: int, fraction: int) -> list[str]:
    """Command-line flags that evaluate one cell under the training protocol."""
    return [*PROTOCOL, "--model", CELLS[(budget, fraction)]]


def cell_outputs() -> list[tuple[tuple[int, int], str]]:
    return [(cell, f"{OUT_DIR}/{cell_name(*cell)}.json") for cell in CELLS]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Restrict to these cell names, e.g. 4h_100 64h_25.")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Evaluate only this many prompts, for a smoke run.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--sbatch-args", action="append", default=[],
                        help="Extra sbatch flags, e.g. cluster partition/gres/time overrides.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    submitted = []
    for cell, out in cell_outputs():
        name = cell_name(*cell)
        if args.cells and name not in args.cells:
            continue
        suffix = "_smoke" if args.num_prompts else ""
        cmd = [
            "sbatch", f"--export=ALL,CELL_NAME={name}{suffix}", *args.sbatch_args,
            SBATCH_SCRIPT,
            "--model-family", MODEL_FAMILY,
            "--merge-root", f"{MERGE_ROOT}/{name}",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--save-generations",
            "--out", out.replace(".json", f"{suffix}.json"),
            *cell_flags(*cell),
        ]
        if args.num_prompts:
            cmd += ["--num-prompts", str(args.num_prompts)]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        submitted.append((name, job_id))
        print(f"{name}: {job_id}")
    if submitted:
        print(f"submitted {len(submitted)} cells")


if __name__ == "__main__":
    main()
