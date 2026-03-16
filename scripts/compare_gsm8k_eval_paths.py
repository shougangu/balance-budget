# ABOUTME: Compares GSM8K scores between eval_strategy (vLLM direct) and lm-eval harness.
# ABOUTME: Reports scores for both the full test set and a configurable subset.

"""
Comparison script: eval_strategy vs lm-eval on GSM8K.

Runs both evaluation paths on the same model, then compares per-question
responses and scoring. Reports scores for both full test set and first N questions.

Usage:
    python scripts/compare_gsm8k_eval_paths.py
"""

import json
import os
import subprocess
import gc
import sys
from contextlib import contextmanager

MAX_TOKENS = 1024
TEMPERATURE = 0.5
SUBSET_SIZE = 571

MODELS = [
    ("llama3-3B", "llama3 base"),
    ("llama3-3B_sft-gsm8k-10000", "llama3 SFT"),
    ("llama3-3B_llama3-3B_gsm8k-p@1-0.3_sft-1024_pt-gsm8k-8976", "llama3 DPO"),
    ("qwen2-2B", "qwen2 base"),
    ("qwen2-2B_sft-gsm8k-10000", "qwen2 SFT"),
    ("qwen2-2B_qwen2-2B_gsm8k-p@1-0.6_sft-1024_pt-gsm8k-8976", "qwen2 DPO"),
]

from tuning.config import set_chat_template, MODELS_DIR

from vllm import LLM, SamplingParams
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.data.config import SYSTEM_MESSAGE_GSM8K
from tuning.training.eval_strategy import GSM8KEvalStrategy
from tuning.evaluation.gsm8k_scoring import (
    is_correct as gsm8k_is_correct,
    extract_gsm8k_answer_strict,
    extract_gsm8k_answer_flexible,
)
from transformers import AutoTokenizer


def run_eval_strategy(model_path: str, output_path: str):
    """Run GSM8K via eval_strategy (vLLM direct). Returns (results_dict, ordered_questions)."""
    print("\n" + "=" * 70)
    print("EVAL_STRATEGY PATH")
    print("=" * 70)

    strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=None)
    messages = strategy.get_test_messages()
    prompts = strategy.get_test_prompts()

    inference_config = VLLMSamplingParamsConfig(
        n=1, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    sampling_params = SamplingParams(**inference_config.model_dump())
    print(f"SamplingParams: {sampling_params}")
    print(f"Num prompts: {len(messages)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_template = tokenizer.chat_template

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
    )

    outputs = llm.chat(messages, sampling_params, chat_template=chat_template)

    results = {}
    ordered_questions = []
    for prompt, msg, output in zip(prompts, messages, outputs):
        response_text = output.outputs[0].text
        user_content = msg[1]["content"]
        raw_question = user_content.split("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0]

        ref = strategy.reference_answers[prompt]
        correct = gsm8k_is_correct(response_text, ref)

        ordered_questions.append(raw_question)
        results[raw_question] = {
            "response": response_text,
            "reference": ref,
            "correct": correct,
            "extracted_strict": extract_gsm8k_answer_strict(response_text),
            "extracted_flexible": extract_gsm8k_answer_flexible(response_text),
        }

    num_correct = sum(r["correct"] for r in results.values())
    total = len(results)
    print(f"\neval_strategy score: {num_correct}/{total} = {num_correct/total:.4f}")

    os.makedirs(output_path, exist_ok=True)
    samples_path = os.path.join(output_path, "samples_eval_strategy.jsonl")
    with open(samples_path, "w") as fh:
        for question in ordered_questions:
            r = results[question]
            fh.write(json.dumps({
                "question": question,
                "response": r["response"],
                "reference": r["reference"],
                "correct": r["correct"],
                "extracted_strict": r["extracted_strict"],
                "extracted_flexible": r["extracted_flexible"],
            }) + "\n")
    print(f"Saved eval_strategy samples to: {samples_path}")

    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return results, ordered_questions


def run_lm_eval(model_path: str, model_name: str) -> dict:
    """Run GSM8K via lm-eval harness. Returns results keyed by question text."""
    print("\n" + "=" * 70)
    print("LM-EVAL PATH")
    print("=" * 70)

    tasks_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tuning", "evaluation", "tasks")
    )

    sampling_params = VLLMSamplingParamsConfig(
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    ).model_dump()

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "gsm8k_eval_compare", model_name
    )
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "lm-eval",
        "--tasks", "gsm8k_matched",
        "--include_path", tasks_dir,
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},max_gen_toks={sampling_params['max_tokens']},enforce_eager=True,gpu_memory_utilization=0.9",
        f"--output_path={output_path}",
        "--log_samples",
        "--batch_size", "auto:N",
        "--gen_kwargs",
        f"temperature={sampling_params['temperature']},top_p={sampling_params['top_p']},top_k={sampling_params['top_k']},skip_special_tokens=true",
        "--apply_chat_template", "true",
        "--system_instruction", SYSTEM_MESSAGE_GSM8K,
    ]

    print(f"Command:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True, text=True, capture_output=True)

    # Find samples JSONL
    samples = None
    for root, _, files in os.walk(output_path):
        for f in files:
            if f.endswith(".jsonl") and "samples" in f:
                path = os.path.join(root, f)
                with open(path) as fh:
                    samples = [json.loads(line) for line in fh]
                print(f"Found {len(samples)} sample entries at: {path}")
                break

    if samples is None:
        raise FileNotFoundError("Could not find lm-eval samples JSONL")

    # Group by question, collect both filter results
    by_question = {}
    for sample in samples:
        doc = sample["doc"]
        question = doc["question"]
        filter_name = sample.get("filter", "")

        if question not in by_question:
            by_question[question] = {
                "ref_answer": doc["answer"].split("####")[-1].strip(),
                "response": sample["resps"][0][0] if sample.get("resps") else "",
            }

        if "strict" in filter_name:
            by_question[question]["lm_eval_strict"] = sample.get("exact_match", 0)
        elif "flexible" in filter_name:
            by_question[question]["lm_eval_flexible"] = sample.get("exact_match", 0)

    # Score lm-eval responses with our scoring
    for data in by_question.values():
        resp = data["response"]
        ref = data["ref_answer"]
        data["our_correct"] = gsm8k_is_correct(resp, ref)

    total = len(by_question)
    our_correct = sum(d["our_correct"] for d in by_question.values())
    lm_strict = sum(d.get("lm_eval_strict", 0) for d in by_question.values())
    lm_flexible = sum(d.get("lm_eval_flexible", 0) for d in by_question.values())

    print(f"\nTotal unique questions: {total}")
    print(f"Our scoring on lm-eval responses: {our_correct}/{total} = {our_correct/total:.4f}")
    print(f"lm-eval strict-match:             {lm_strict}/{total} = {lm_strict/total:.4f}")
    print(f"lm-eval flexible-extract:         {lm_flexible}/{total} = {lm_flexible/total:.4f}")

    return by_question


def score_subset(eval_strat, lm_eval, questions):
    """Compute scores for a subset of questions present in both result sets."""
    common = [q for q in questions if q in eval_strat and q in lm_eval]
    n = len(common)
    if n == 0:
        return None

    es_correct = sum(eval_strat[q]["correct"] for q in common)
    le_ours = sum(lm_eval[q]["our_correct"] for q in common)
    le_strict = sum(lm_eval[q].get("lm_eval_strict", 0) for q in common)
    le_flex = sum(lm_eval[q].get("lm_eval_flexible", 0) for q in common)

    # Count response differences
    different = sum(1 for q in common if eval_strat[q]["response"] != lm_eval[q]["response"])

    return {
        "n": n,
        "eval_strategy": es_correct / n,
        "lm_eval_ours": le_ours / n,
        "lm_eval_strict": le_strict / n,
        "lm_eval_flexible": le_flex / n,
        "delta_pp": (es_correct / n - le_ours / n) * 100,
        "response_diff_pct": different / n * 100,
    }


def print_summary_table(title, summaries):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    header = f"{'Model':<16} {'N':>5} {'eval_strat':>11} {'lm-eval(ours)':>14} {'lm strict':>10} {'lm flex':>10} {'Delta(pp)':>10} {'Resp diff':>10}"
    print(header)
    print("-" * 110)
    for label, s in summaries.items():
        if s is None:
            print(f"{label:<16}  (no data)")
            continue
        print(f"{label:<16} {s['n']:>5} {s['eval_strategy']:>10.2%} {s['lm_eval_ours']:>13.2%} {s['lm_eval_strict']:>9.2%} {s['lm_eval_flexible']:>9.2%} {s['delta_pp']:>+9.2f} {s['response_diff_pct']:>9.1f}%")


if __name__ == "__main__":
    summaries_full = {}
    summaries_subset = {}

    for model_name, label in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"Model not found, skipping: {model_path}")
            continue

        print(f"\n{'#' * 70}")
        print(f"# MODEL: {label} ({model_name})")
        print(f"# Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
        print(f"{'#' * 70}")

        set_chat_template(model_name)

        output_path = os.path.join(
            os.path.dirname(__file__), "..", "artifacts", "gsm8k_eval_compare", model_name
        )
        eval_strat, ordered_questions = run_eval_strategy(model_path, output_path)
        lm_eval_results = run_lm_eval(model_path, model_name)

        summaries_full[label] = score_subset(eval_strat, lm_eval_results, ordered_questions)
        summaries_subset[label] = score_subset(eval_strat, lm_eval_results, ordered_questions[:SUBSET_SIZE])

    print_summary_table(f"FULL TEST SET", summaries_full)
    print_summary_table(f"FIRST {SUBSET_SIZE} QUESTIONS", summaries_subset)
