# ABOUTME: One-off quality audit for GSM8K SFT and DPO datasets as loaded by the training pipeline.
# ABOUTME: Checks unique prompts, duplicates, answer correctness, and structural issues at each stage.

"""
Run: python -m tests.check_gsm8k_quality
"""

from collections import Counter
from datasets import load_from_disk
from tuning.evaluation.gsm8k_scoring import is_correct, extract_gsm8k_answer_strict, extract_gsm8k_answer_flexible
from tuning.config import DATASETS_DIR
from tuning.data.utils import get_random_train_subset


def print_section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)


def audit_sft_saved():
    """Audit the saved sft-gsm8k dataset (after _get_rows expansion + train_test_split)."""
    print_section("SAVED SFT DATASET: sft-gsm8k")
    path = f"{DATASETS_DIR}/sft-gsm8k"
    ds = load_from_disk(path)
    print(f"Loaded from: {path}")
    print(f"Splits: train={len(ds['train'])}, test={len(ds['test'])}")
    print(f"Columns: {ds['train'].column_names}")

    train = ds["train"]

    # Unique prompts
    prompts = train["prompt"]
    unique_prompts = set(prompts)
    print(f"\nUnique prompts (train): {len(unique_prompts)}")
    print(f"Total rows (train): {len(prompts)}")
    print(f"Avg rows per prompt: {len(prompts) / len(unique_prompts):.1f}")

    prompt_counts = Counter(prompts)
    print(f"\nPrompt frequency distribution:")
    freq_dist = Counter(prompt_counts.values())
    for count, num_prompts in sorted(freq_dist.items()):
        print(f"  {count} rows: {num_prompts} prompts")

    # Correctness - SFT has messages, extract assistant content and check against...
    # SFT doesn't store reference_answer, so we check if answers have #### format
    sample_msgs = train[0]["messages"]
    print(f"\nSample message structure (row 0):")
    for msg in sample_msgs:
        print(f"  {msg['role']}: {msg['content'][:100]}...")

    # Check #### extraction works on all assistant responses
    no_extract = 0
    for i in range(len(train)):
        assistant_content = train[i]["messages"][-1]["content"]
        strict = extract_gsm8k_answer_strict(assistant_content)
        flex = extract_gsm8k_answer_flexible(assistant_content)
        if strict is None and flex is None:
            no_extract += 1
            if no_extract <= 3:
                print(f"\n  WARNING: No answer extractable from row {i}:")
                print(f"    {assistant_content[:200]}")

    print(f"\nAnswer extraction: {len(train) - no_extract}/{len(train)} extractable, {no_extract} failures")

    # Now simulate get_random_train_subset at a typical size
    for size in [5000, 10000]:
        if size > len(train):
            print(f"\n[Subset {size}] Skipped - larger than train ({len(train)})")
            continue
        subset = get_random_train_subset(ds, size, unique_column="prompt")
        subset_prompts = subset["train"]["prompt"]
        subset_unique = len(set(subset_prompts))
        print(f"\n[Subset {size}] {len(subset['train'])} rows, {subset_unique} unique prompts "
              f"({100*subset_unique/len(subset['train']):.1f}% unique)")


def audit_dpo_saved():
    """Audit the saved pt-gsm8k dataset (after filter_correct_chosen + sample_correct_chosen)."""
    print_section("SAVED DPO DATASET: pt-gsm8k")
    path = f"{DATASETS_DIR}/pt-gsm8k"
    ds = load_from_disk(path)
    print(f"Loaded from: {path}")
    print(f"Splits: train={len(ds['train'])}, test={len(ds['test'])}")
    print(f"Columns: {ds['train'].column_names}")

    train = ds["train"]

    # Unique prompts
    prompts = train["prompt"]
    unique_prompts = set(prompts)
    print(f"\nUnique prompts (train): {len(unique_prompts)}")
    print(f"Total rows (train): {len(prompts)}")
    print(f"Avg rows per prompt: {len(prompts) / len(unique_prompts):.1f}")

    prompt_counts = Counter(prompts)
    print(f"\nPrompt frequency distribution:")
    freq_dist = Counter(prompt_counts.values())
    for count, num_prompts in sorted(freq_dist.items()):
        print(f"  {count} rows: {num_prompts} prompts")

    # Sample row structure
    print(f"\nSample row (row 0):")
    row0 = train[0]
    for key in row0:
        val = str(row0[key])
        print(f"  {key}: {val[:120]}{'...' if len(val) > 120 else ''}")

    # Chosen correctness (should all be correct after filter_correct_chosen)
    chosen_correct = 0
    chosen_incorrect = 0
    chosen_incorrect_examples = []
    for i in range(len(train)):
        ref = train[i]["reference_solution"]
        chosen = train[i]["chosen"]
        if is_correct(chosen, ref):
            chosen_correct += 1
        else:
            chosen_incorrect += 1
            if len(chosen_incorrect_examples) < 3:
                chosen_incorrect_examples.append((i, ref, chosen[:150]))

    print(f"\nChosen correctness (should be 100% after filtering):")
    print(f"  Correct: {chosen_correct} ({100*chosen_correct/len(train):.1f}%)")
    print(f"  Incorrect: {chosen_incorrect} ({100*chosen_incorrect/len(train):.1f}%)")
    if chosen_incorrect_examples:
        print(f"  PROBLEM - incorrect chosen survived filtering:")
        for idx, ref, preview in chosen_incorrect_examples:
            print(f"    Row {idx}: ref={ref}, chosen={preview}")

    # Rejected correctness (ideally all wrong, but some may be correct)
    rejected_correct = 0
    both_correct = 0
    both_incorrect = 0
    for i in range(len(train)):
        ref = train[i]["reference_solution"]
        c = is_correct(train[i]["chosen"], ref)
        r = is_correct(train[i]["rejected"], ref)
        if r:
            rejected_correct += 1
        if c and r:
            both_correct += 1
        if not c and not r:
            both_incorrect += 1

    print(f"\nRejected correctness:")
    print(f"  Incorrect (good): {len(train) - rejected_correct} ({100*(len(train)-rejected_correct)/len(train):.1f}%)")
    print(f"  Correct (no DPO signal): {rejected_correct} ({100*rejected_correct/len(train):.1f}%)")

    print(f"\nPair quality:")
    ideal = chosen_correct - both_correct
    print(f"  Chosen correct, rejected incorrect (ideal): {ideal} ({100*ideal/len(train):.1f}%)")
    print(f"  Both correct (no signal): {both_correct} ({100*both_correct/len(train):.1f}%)")
    print(f"  Both incorrect (no signal): {both_incorrect} ({100*both_incorrect/len(train):.1f}%)")

    # Simulate get_random_train_subset at typical sizes
    for size in [2500, 5000, 7500]:
        if size > len(train):
            print(f"\n[Subset {size}] Skipped - larger than train ({len(train)})")
            continue
        subset = get_random_train_subset(ds, size, unique_column="prompt")
        subset_train = subset["train"]
        subset_prompts = subset_train["prompt"]
        subset_unique = len(set(subset_prompts))
        # Check pair quality of subset
        sub_both_correct = 0
        sub_ideal = 0
        for j in range(len(subset_train)):
            ref = subset_train[j]["reference_solution"]
            c = is_correct(subset_train[j]["chosen"], ref)
            r = is_correct(subset_train[j]["rejected"], ref)
            if c and not r:
                sub_ideal += 1
            if c and r:
                sub_both_correct += 1
        print(f"\n[Subset {size}] {len(subset_train)} rows, {subset_unique} unique prompts, "
              f"{sub_ideal} ideal pairs ({100*sub_ideal/len(subset_train):.1f}%), "
              f"{sub_both_correct} both-correct ({100*sub_both_correct/len(subset_train):.1f}%)")


def audit_test_splits():
    """Check that SFT and DPO test splits are reasonable."""
    print_section("TEST SPLIT COMPARISON")

    sft_ds = load_from_disk(f"{DATASETS_DIR}/sft-gsm8k")
    dpo_ds = load_from_disk(f"{DATASETS_DIR}/pt-gsm8k")

    sft_test = sft_ds["test"]
    dpo_test = dpo_ds["test"]

    print(f"SFT test size: {len(sft_test)}")
    print(f"DPO test size: {len(dpo_test)}")

    # Check if they share the same prompts (they shouldn't if created from different HF datasets)
    sft_has_prompt = "prompt" in sft_test.column_names
    dpo_has_prompt = "prompt" in dpo_test.column_names

    if sft_has_prompt:
        sft_test_prompts = set(sft_test["prompt"])
        print(f"SFT test unique prompts: {len(sft_test_prompts)}")
    if dpo_has_prompt:
        dpo_test_prompts = set(dpo_test["prompt"])
        print(f"DPO test unique prompts: {len(dpo_test_prompts)}")
    if sft_has_prompt and dpo_has_prompt:
        overlap = sft_test_prompts & dpo_test_prompts
        print(f"Prompt overlap between SFT/DPO test: {len(overlap)}")


if __name__ == "__main__":
    audit_sft_saved()
    audit_dpo_saved()
    audit_test_splits()
