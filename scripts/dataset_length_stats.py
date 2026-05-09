# ABOUTME: Reports # of train prompts, mean/median prompt length, and mean/median completion length
# ABOUTME: for simpleRL/openmath/gsm8k RLVR + SFT datasets, tokenized with the llama3 tokenizer.

import random
import statistics
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from tuning.utils.utils import LLAMA_31_SIMPLE_TEMPLATE

DATASETS_DIR = Path("/project/6105902/shougan/balance-budget/tuning/data/datasets")
TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
SAMPLE_CAP = 50_000  # rows to sample for very large datasets
TOKENIZE_BATCH = 1024
SEED = 0

RLVR_DATASETS = [
    "rlvr-simplerl-easy",
    "rlvr-simplerl-medium",
    "rlvr-simplerl-hard",
    "rlvr-openmath",
    "rlvr-gsm8k",
]

SFT_DATASETS = [
    "sft-openmath",
    "sft-gsm8k",
]


def stats(values):
    sorted_v = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": sorted_v[int(0.95 * (len(sorted_v) - 1))],
        "max": max(values),
    }


def maybe_subsample(ds, total):
    if total <= SAMPLE_CAP:
        return ds, total, False
    rng = random.Random(SEED)
    idx = rng.sample(range(total), SAMPLE_CAP)
    idx.sort()
    return ds.select(idx), SAMPLE_CAP, True


def tokenize_lengths(tokenizer, texts):
    lengths = []
    for i in range(0, len(texts), TOKENIZE_BATCH):
        chunk = texts[i : i + TOKENIZE_BATCH]
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in enc)
    return lengths


def measure_rlvr(name, tokenizer):
    full = load_from_disk(str(DATASETS_DIR / name))["train"]
    n_total = len(full)
    sub, n_sampled, sampled = maybe_subsample(full, n_total)
    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in sub["prompt"]
    ]
    answer_texts = list(sub["reference_answer"])
    p_lens = tokenize_lengths(tokenizer, prompt_texts)
    a_lens = tokenize_lengths(tokenizer, answer_texts)
    return n_total, n_sampled, sampled, stats(p_lens), stats(a_lens)


def measure_sft(name, tokenizer):
    full = load_from_disk(str(DATASETS_DIR / name))["train"]
    n_total = len(full)
    sub, n_sampled, sampled = maybe_subsample(full, n_total)
    prompt_texts = []
    completion_texts = []
    for msgs in sub["messages"]:
        non_assistant = [m for m in msgs if m["role"] != "assistant"]
        assistant = next(m for m in msgs if m["role"] == "assistant")
        prompt_texts.append(
            tokenizer.apply_chat_template(
                non_assistant, tokenize=False, add_generation_prompt=True
            )
        )
        completion_texts.append(assistant["content"])
    p_lens = tokenize_lengths(tokenizer, prompt_texts)
    c_lens = tokenize_lengths(tokenizer, completion_texts)
    return n_total, n_sampled, sampled, stats(p_lens), stats(c_lens)


def fmt(s):
    return (
        f"mean={s['mean']:>7.1f}  median={s['median']:>6}  "
        f"p95={s['p95']:>6}  max={s['max']:>6}"
    )


def header(name, n_total, n_sampled, sampled):
    suffix = f" (sampled {n_sampled:,} of {n_total:,})" if sampled else ""
    return f"[{name}]  prompts={n_total:,}{suffix}"


def main():
    print(f"Tokenizer: {TOKENIZER_NAME} (with LLAMA_31_SIMPLE_TEMPLATE)")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.chat_template = LLAMA_31_SIMPLE_TEMPLATE

    print(
        "\n=== RLVR (prompt = chat template + generation prompt; "
        "completion = reference_answer string) ==="
    )
    for name in RLVR_DATASETS:
        n_total, n_sampled, sampled, p, c = measure_rlvr(name, tokenizer)
        print(f"\n{header(name, n_total, n_sampled, sampled)}")
        print(f"  prompt    : {fmt(p)}")
        print(f"  reference : {fmt(c)}")

    print(
        "\n=== SFT (prompt = system+user chat template + generation prompt; "
        "completion = assistant message) ==="
    )
    for name in SFT_DATASETS:
        n_total, n_sampled, sampled, p, c = measure_sft(name, tokenizer)
        print(f"\n{header(name, n_total, n_sampled, sampled)}")
        print(f"  prompt    : {fmt(p)}")
        print(f"  completion: {fmt(c)}")


if __name__ == "__main__":
    main()
