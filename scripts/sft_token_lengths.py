# ABOUTME: Measures the mean tokenized (truncated) length of an SFT example per model,
# ABOUTME: chat template and dataset, so SFT compute can be expressed in FLOPs.

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from tuning.config import DATASETS_DIR, HF_MODEL_MAP
from tuning.utils.utils import GEMMA_3_CHAT_TEMPLATE, LLAMA_31_SIMPLE_TEMPLATE, SIMPLE_TEMPLATE

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "budget_frontier" / "sft_token_lengths.json"

# Rendered chat templates, exactly the strings the training tokenizer ends up with
# after chat_template_func; unsloth's setup step is not needed to render text.
TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "llama-3.1": LLAMA_31_SIMPLE_TEMPLATE,
    "gemma-3": GEMMA_3_CHAT_TEMPLATE,
}

# (model, template, dataset, max_length) combinations used by the budget-split campaign.
SPECS = [
    ("llama3-3B", "simple", "sft-openmath", 1024),
    ("llama3-8B", "simple", "sft-openmath", 1024),
    ("gemma3-4B", "simple", "sft-openmath", 1024),
    ("gemma3-12B", "simple", "sft-openmath", 1024),
    ("gemma3-12B", "gemma-3", "sft-ifmix", 4096),
]

SAMPLE_ROWS = 20_000
SEED = 0
TOKENIZE_BATCH = 512


def spec_key(model, template, dataset, max_length):
    return f"{model}|{template}|{dataset}|{max_length}"


def mean_example_tokens(tokenizer, conversations, max_length):
    """Mean number of tokens per rendered conversation after truncation to max_length."""
    total = 0
    for start in range(0, len(conversations), TOKENIZE_BATCH):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in conversations[start : start + TOKENIZE_BATCH]
        ]
        encoded = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False, add_special_tokens=False,
        )["input_ids"]
        total += sum(len(ids) for ids in encoded)
    return total / len(conversations)


def sample_conversations(dataset_name, rows, seed):
    train = load_from_disk(f"{DATASETS_DIR}/{dataset_name}")["train"]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(train)), min(rows, len(train))))
    return train.select(idx)["messages"], len(train)


def measure(specs, rows=SAMPLE_ROWS, seed=SEED):
    results = {}
    samples = {}
    for model, template, dataset, max_length in specs:
        if dataset not in samples:
            samples[dataset] = sample_conversations(dataset, rows, seed)
        conversations, dataset_rows = samples[dataset]
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_MAP[model])
        tokenizer.chat_template = TEMPLATES[template]
        mean_tokens = mean_example_tokens(tokenizer, conversations, max_length)
        results[spec_key(model, template, dataset, max_length)] = {
            "model": model,
            "template": template,
            "dataset": dataset,
            "max_length": max_length,
            "mean_tokens": mean_tokens,
            "sampled_rows": len(conversations),
            "dataset_rows": dataset_rows,
        }
        print(f"{model:11s} {template:9s} {dataset:13s} max={max_length:5d}  mean tokens/example = {mean_tokens:.1f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=SAMPLE_ROWS)
    args = parser.parse_args()
    results = measure(SPECS, rows=args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
