# ABOUTME: Batch IFeval harness that reads model entries from metadata JSONL files.
# ABOUTME: Runs 3 inference+eval iterations per model, writes strict/loose scores to results.md.

from tuning.evaluation.ifeval_evaluate import ifeval_evaluate
from tuning.inference.ifeval_inference import run_inference_ifeval
from tuning.config import MODELS_METADATA_DIR, set_chat_template
from pathlib import Path
from statistics import mean
import json
import re


def parse_ifeval_stdout(stdout_text):
    """Extract strict/loose prompt-level and instruction-level scores from eval stdout.

    The output contains two sections (strict first, loose second), each with
    a prompt-level and instruction-level line. We grab all occurrences via findall
    and map them positionally: first pair = strict, second pair = loose.
    """
    prompts = re.findall(r'prompt-level:\s*([\d.]+)', stdout_text)
    instructions = re.findall(r'instruction-level:\s*([\d.]+)', stdout_text)

    if len(prompts) < 2 or len(instructions) < 2:
        raise ValueError(f"Unable to parse IFeval metrics from stdout: "
                         f"found {len(prompts)} prompt-level, {len(instructions)} instruction-level")

    return {
        "strict": {
            "prompt_level": float(prompts[0]),
            "instruction_level": float(instructions[0]),
        },
        "loose": {
            "prompt_level": float(prompts[1]),
            "instruction_level": float(instructions[1]),
        },
}

METADATA_FILES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5"][::-1]
TOTAL_SIZE = 10000
NUM_RUNS = 1
RESULTS_PATH = f"{MODELS_METADATA_DIR}/results_dpo_chat_template_reverse.md"


def format_model_results(model_name, all_scores):
    """Format per-run scores and averages as a markdown table for one model."""
    lines = []
    lines.append(f"\n## {model_name}\n")
    lines.append("| Run | Prompt (Strict) | Instruction (Strict) | Prompt (Loose) | Instruction (Loose) |")
    lines.append("|-----|-----------------|----------------------|----------------|---------------------|")
    for i, scores in enumerate(all_scores, 1):
        lines.append(
            f"| {i} "
            f"| {scores['strict']['prompt_level']:.4f} "
            f"| {scores['strict']['instruction_level']:.4f} "
            f"| {scores['loose']['prompt_level']:.4f} "
            f"| {scores['loose']['instruction_level']:.4f} |"
        )
    avg_sp = mean(s["strict"]["prompt_level"] for s in all_scores)
    avg_si = mean(s["strict"]["instruction_level"] for s in all_scores)
    avg_lp = mean(s["loose"]["prompt_level"] for s in all_scores)
    avg_li = mean(s["loose"]["instruction_level"] for s in all_scores)
    lines.append(f"| Avg | {avg_sp:.4f} | {avg_si:.4f} | {avg_lp:.4f} | {avg_li:.4f} |")
    return "\n".join(lines) + "\n"


def evaluate_model(model_name, results_file):
    """Run inference and evaluation for a single model, write results."""
    if "llama" in model_name:
        set_chat_template("llama3-1B")
    else:
        set_chat_template("qwen2-3B")   
    all_scores = []
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"Running inference for {model_name} (run {run_idx}/{NUM_RUNS})")
        run_inference_ifeval(model_name, n_samples=1, temperature=0.5, save_results=True)

        print(f"Running evaluation for {model_name} (run {run_idx}/{NUM_RUNS})")
        stdout_text = ifeval_evaluate(model_name)
        scores = parse_ifeval_stdout(stdout_text)
        all_scores.append(scores)

    result_text = format_model_results(model_name, all_scores)
    results_file.write(result_text)
    results_file.flush()
    print(f"Wrote results for {model_name}")


def pt_model_name_from_entry(entry):
    """Derive the PT counterpart model name from an SFT metadata entry."""
    sft_name = Path(entry["checkpoint_path"]).name
    base_model = sft_name.split("_", 1)[0]
    remaining_budget = TOTAL_SIZE - entry["data_points_seen"]
    return f"{base_model}_{sft_name}_pt-tuluif-{remaining_budget}"


if __name__ == "__main__":
    with open(RESULTS_PATH, "a") as results_file:
        for entry_name in METADATA_FILES:
            entry_path = f"{MODELS_METADATA_DIR}/{entry_name}.json"
            with open(entry_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if not entry.get("completed"):
                        print(f"Skipping {entry['checkpoint_path']} since it's not completed.")
                        continue

                    pt_name = pt_model_name_from_entry(entry)
                    evaluate_model(pt_name, results_file)
