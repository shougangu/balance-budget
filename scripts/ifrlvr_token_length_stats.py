# ABOUTME: Tokenizes IF-RLVR prompts (system + user, with each model's chat template) using the
# ABOUTME: llama3, qwen2.5, and gemma-3 tokenizers and reports how a 1024-token prompt budget filters them.

import statistics

from huggingface_hub import hf_hub_download
import pandas as pd
from transformers import AutoTokenizer

from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
from tuning.utils.utils import LLAMA_31_SIMPLE_TEMPLATE, GEMMA_3_CHAT_TEMPLATE

HF_PATH = "allenai/IF_multi_constraints_upto5"
MAX_PROMPT_TOKENS = 1024
TOKENIZE_BATCH = 1024

# (label, hf model, chat template override or None to use the tokenizer's built-in template)
TOKENIZERS = [
    ("llama3", "unsloth/Meta-Llama-3.1-8B", LLAMA_31_SIMPLE_TEMPLATE),
    ("qwen2.5", "unsloth/Qwen2.5-7B-Instruct", None),  # ships chatml; same vocab as base
    ("gemma-3", "unsloth/gemma-3-4b-pt", GEMMA_3_CHAT_TEMPLATE),
]


def load_prompts():
    parquet_path = hf_hub_download(
        HF_PATH, "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    seen = set()
    prompts = []
    for content in df["messages"]:
        user_text = content[0]["content"]
        if user_text in seen:
            continue
        seen.add(user_text)
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                {"role": "user", "content": user_text},
            ]
        )
    return prompts


def prompt_token_lengths(tokenizer, prompts):
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    lengths = []
    for i in range(0, len(texts), TOKENIZE_BATCH):
        chunk = texts[i : i + TOKENIZE_BATCH]
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in enc)
    return lengths


def stats(values):
    sorted_v = sorted(values)

    def pct(p):
        return sorted_v[int(p * (len(sorted_v) - 1))]

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "p99.9": pct(0.999),
        "max": max(values),
    }


def main():
    prompts = load_prompts()
    n = len(prompts)
    print(f"IF-RLVR: {n:,} unique prompts (system + user, with generation prompt)\n")
    print(f"Prompt-token budget under test: {MAX_PROMPT_TOKENS}\n")

    for label, model, template in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(model)
        if template is not None:
            tok.chat_template = template
        lengths = prompt_token_lengths(tok, prompts)
        s = stats(lengths)
        over = sum(1 for v in lengths if v > MAX_PROMPT_TOKENS)
        print(f"[{label}]  ({model})")
        print(
            f"  mean={s['mean']:>7.1f}  median={s['median']:>5}  p95={s['p95']:>5}  "
            f"p99={s['p99']:>5}  p99.9={s['p99.9']:>6}  max={s['max']:>6}"
        )
        print(
            f"  > {MAX_PROMPT_TOKENS} tokens: {over:,} / {n:,} "
            f"({100 * over / n:.2f}%) dropped\n"
        )


if __name__ == "__main__":
    main()
