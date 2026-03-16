# ABOUTME: Runs both lm-eval and eval_strategy paths on the same model to confirm equivalence.
# ABOUTME: Saves per-question results and scores to outputs/live_lm_comparison/.

"""
Comparison script: lm-eval harness vs eval_strategy (direct vLLM) on GSM8K.

Runs both evaluation paths on the same model with identical config, then
compares per-question responses and scoring to confirm they are equivalent.
Uses PassAtKStoppingCallback's inference methods for the eval_strategy path.

Usage:
    python scripts/live_lm_comparison.py model_name [model_name2 ...]
    python scripts/live_lm_comparison.py --temperature 0.0 --max-tokens 1024 qwen2-2B llama3-3B
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from tuning.config import MODELS_DIR, OUTPUTS_DIR, set_chat_template
from tuning.evaluation.gsm8k_evaluate import gsm8k_evaluate
from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct
from tuning.training.config_training import PassAtKConfig
from tuning.training.eval_strategy import GSM8KEvalStrategy
from tuning.training.passk_callback import PassAtKStoppingCallback

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "live_lm_comparison")


def run_eval_strategy_path(model_name, model_path, temperature, max_tokens,
                           gpu_memory_utilization, output_dir):
    """Run GSM8K via eval_strategy using PassAtKStoppingCallback's vLLM inference."""
    print("\n" + "=" * 70)
    print("EVAL_STRATEGY PATH (via PassAtKStoppingCallback)")
    print("=" * 70)

    strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = PassAtKConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        use_persistent_vllm=False,
        vllm_gpu_memory_utilization=gpu_memory_utilization,
    )

    callback = PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name=model_name,
        base_model_hf=model_path,
        primary_eval=strategy,
    )

    # Use callback's ephemeral vLLM methods for full-model inference (no LoRA adapter)
    llm = callback._create_ephemeral_vllm()
    model_results = callback._run_vllm_inference(llm, strategy, adapter_path=None)
    callback._cleanup_ephemeral_vllm(llm)

    scores = strategy.score_responses(model_results, tokenizer)

    # Build per-question detail for comparison
    per_question = {}
    for item in model_results:
        prompt = item["prompt"]
        response = item["responses"][0]
        ref = strategy.reference_answers[prompt]
        correct = gsm8k_is_correct(response, ref)
        per_question[prompt] = {
            "response": response,
            "reference": ref,
            "correct": correct,
        }

    num_correct = sum(r["correct"] for r in per_question.values())
    total = len(per_question)
    print(f"\neval_strategy score: {num_correct}/{total} = {num_correct/total:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_strategy_results.jsonl"), "w") as fh:
        for prompt, detail in per_question.items():
            fh.write(json.dumps({"prompt": prompt, **detail}) + "\n")
    with open(os.path.join(output_dir, "eval_strategy_scores.json"), "w") as fh:
        json.dump(
            {k: float(v) if isinstance(v, (float, int)) else v for k, v in scores.items()},
            fh, indent=2,
        )

    return scores, per_question


def run_lm_eval_path(model_name, temperature, max_tokens, gpu_memory_utilization,
                     output_dir):
    """Run GSM8K via lm-eval harness. Returns per-question results keyed by prompt."""
    print("\n" + "=" * 70)
    print("LM-EVAL PATH")
    print("=" * 70)

    lm_eval_output = gsm8k_evaluate(
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    if lm_eval_output is None:
        print("lm-eval failed, skipping comparison")
        return None

    # Find the most recent lm-eval samples file
    sample_files = []
    for root, _, files in os.walk(lm_eval_output):
        for f in files:
            if f.endswith(".jsonl") and "samples" in f:
                sample_files.append(os.path.join(root, f))

    if not sample_files:
        print("Could not find lm-eval samples JSONL")
        return None

    path = max(sample_files, key=os.path.getmtime)
    with open(path) as fh:
        samples = [json.loads(line) for line in fh]
    print(f"Found {len(samples)} sample entries at: {path}")


    # Build per-question results from lm-eval output
    from tuning.data.config import GSM8K_STRING
    per_question = {}
    for sample in samples:
        doc = sample["doc"]
        question = doc["question"]
        prompt = GSM8K_STRING.format(question=question)
        filter_name = sample.get("filter", "")

        if prompt not in per_question:
            ref_answer = doc["answer"].split("####")[-1].strip()
            response = sample["resps"][0][0] if sample.get("resps") else ""
            our_correct = gsm8k_is_correct(response, ref_answer)
            per_question[prompt] = {
                "response": response,
                "reference": ref_answer,
                "correct": our_correct,
            }

        if "strict" in filter_name:
            per_question[prompt]["lm_eval_strict"] = sample.get("exact_match", 0)
        elif "flexible" in filter_name:
            per_question[prompt]["lm_eval_flexible"] = sample.get("exact_match", 0)

    total = len(per_question)
    our_correct = sum(d["correct"] for d in per_question.values())
    print(f"\nlm-eval path (our scoring): {our_correct}/{total} = {our_correct/total:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "lm_eval_results.jsonl"), "w") as fh:
        for prompt, detail in per_question.items():
            fh.write(json.dumps({"prompt": prompt, **detail}) + "\n")

    return per_question


def compare_results(eval_strat, lm_eval, output_dir):
    """Compare per-question results between eval_strategy and lm-eval paths."""
    common_prompts = [p for p in eval_strat if p in lm_eval]
    n = len(common_prompts)

    if n == 0:
        print("No common prompts found between the two paths!")
        return None

    es_correct = sum(eval_strat[p]["correct"] for p in common_prompts)
    le_correct = sum(lm_eval[p]["correct"] for p in common_prompts)
    response_diffs = sum(
        1 for p in common_prompts
        if eval_strat[p]["response"] != lm_eval[p]["response"]
    )
    scoring_diffs = sum(
        1 for p in common_prompts
        if eval_strat[p]["correct"] != lm_eval[p]["correct"]
    )

    summary = {
        "total_prompts": n,
        "eval_strategy_correct": es_correct,
        "lm_eval_correct": le_correct,
        "eval_strategy_accuracy": es_correct / n,
        "lm_eval_accuracy": le_correct / n,
        "response_differences": response_diffs,
        "scoring_differences": scoring_diffs,
        "delta_pp": (es_correct / n - le_correct / n) * 100,
    }

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Common prompts:       {n}")
    print(f"eval_strategy:        {es_correct}/{n} = {es_correct/n:.4f}")
    print(f"lm-eval (our score):  {le_correct}/{n} = {le_correct/n:.4f}")
    print(f"Delta:                {summary['delta_pp']:+.2f} pp")
    print(f"Response differences: {response_diffs}/{n} ({response_diffs/n*100:.1f}%)")
    print(f"Scoring differences:  {scoring_diffs}/{n} ({scoring_diffs/n*100:.1f}%)")

    if scoring_diffs > 0:
        print(f"\nScoring disagreements (first 10):")
        count = 0
        for p in common_prompts:
            if eval_strat[p]["correct"] != lm_eval[p]["correct"]:
                print(f"  prompt: {p[:60]}...")
                print(f"    eval_strategy: correct={eval_strat[p]['correct']}")
                print(f"    lm-eval:       correct={lm_eval[p]['correct']}")
                count += 1
                if count >= 10:
                    break

    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def evaluate_model(model_name, temperature, max_tokens, gpu_memory_utilization):
    """Run both evaluation paths on a model and compare."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found, skipping: {model_path}")
        return None

    set_chat_template(model_name)
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)

    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"# Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"{'#'*70}")

    # Run eval_strategy path (PassAtKStoppingCallback + vLLM)
    _, es_results = run_eval_strategy_path(
        model_name, model_path, temperature, max_tokens,
        gpu_memory_utilization, model_output_dir,
    )

    # Run lm-eval path
    le_results = run_lm_eval_path(
        model_name, temperature, max_tokens, gpu_memory_utilization, model_output_dir,
    )

    if le_results is None:
        print("Skipping comparison due to lm-eval failure")
        return None

    # Compare
    return compare_results(es_results, le_results, model_output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare lm-eval and eval_strategy GSM8K evaluation paths",
    )
    parser.add_argument("models", nargs="+", help="Model names to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (use 0.0 for deterministic comparison)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    args = parser.parse_args()

    summaries = {}
    for model_name in args.models:
        summary = evaluate_model(
            model_name, args.temperature, args.max_tokens, args.gpu_mem,
        )
        if summary:
            summaries[model_name] = summary

    if summaries:
        print(f"\n\n{'='*90}")
        print("OVERALL COMPARISON")
        print(f"{'='*90}")
        header = (f"{'Model':<30} {'N':>5} {'eval_strat':>11} {'lm-eval':>11} "
                  f"{'Delta(pp)':>10} {'Score diffs':>11}")
        print(header)
        print("-" * 90)
        for model, s in summaries.items():
            print(f"{model:<30} {s['total_prompts']:>5} "
                  f"{s['eval_strategy_accuracy']:>10.2%} "
                  f"{s['lm_eval_accuracy']:>10.2%} "
                  f"{s['delta_pp']:>+9.2f} "
                  f"{s['scoring_differences']:>10}")


if __name__ == "__main__":
    main()
