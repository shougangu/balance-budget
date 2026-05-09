# ABOUTME: Token-length stats for open-r1/OpenR1-Math-220k (default config) using llama3 tokenizer.
# ABOUTME: Mirrors scripts/dataset_length_stats.py: prompt = chat template + generation prompt,
# ABOUTME: completion = assistant message (R1-distilled long CoT).

import random
import statistics

from datasets import load_dataset
from transformers import AutoTokenizer

from tuning.utils.utils import LLAMA_31_SIMPLE_TEMPLATE

HF_PATH = "open-r1/OpenR1-Math-220k"
CONFIGS = ["default", "extended", "all"]
TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
SAMPLE_CAP = 50_000
TOKENIZE_BATCH = 256
SEED = 0


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


def fmt(s):
    return (
        f"mean={s['mean']:>8.1f}  median={s['median']:>7}  "
        f"p95={s['p95']:>7}  max={s['max']:>7}"
    )


def measure(config, tokenizer):
    full = load_dataset(HF_PATH, config, split="train")
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


def main():
    print(f"Tokenizer: {TOKENIZER_NAME} (with LLAMA_31_SIMPLE_TEMPLATE)")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.chat_template = LLAMA_31_SIMPLE_TEMPLATE

    for cfg in CONFIGS:
        n_total, n_sampled, sampled, p, c = measure(cfg, tokenizer)
        suffix = f" (sampled {n_sampled:,} of {n_total:,})" if sampled else ""
        print(f"\n[{HF_PATH}::{cfg}]  prompts={n_total:,}{suffix}")
        print(f"  prompt    : {fmt(p)}")
        print(f"  completion: {fmt(c)}")


if __name__ == "__main__":
    main()
