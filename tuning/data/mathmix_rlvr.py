# ABOUTME: Builds the 50/50 SimpleRL+OpenMath mixed RLVR dataset for GRPO training.
# ABOUTME: Full rlvr-simplerl train + an equal-size dedup'd sample of rlvr-openmath.

import random

from datasets import Dataset, DatasetDict, load_from_disk

from tuning.config import DATASETS_DIR


def build_mathmix(simplerl: DatasetDict, openmath: DatasetDict, seed: int = 42) -> DatasetDict:
    """Return a DatasetDict whose train split is all simplerl rows plus an
    equal number of openmath rows (prompts already in simplerl are skipped),
    shuffled together. The test split takes half from each source's test."""
    rng = random.Random(seed)

    simplerl_rows = list(simplerl["train"])
    seen = {r["prompt"][-1]["content"] for r in simplerl_rows}

    openmath_indices = list(range(openmath["train"].num_rows))
    rng.shuffle(openmath_indices)
    openmath_rows = []
    for i in openmath_indices:
        if len(openmath_rows) == len(simplerl_rows):
            break
        row = openmath["train"][i]
        key = row["prompt"][-1]["content"]
        if key in seen:
            continue
        seen.add(key)
        openmath_rows.append(row)

    train_rows = simplerl_rows + openmath_rows
    rng.shuffle(train_rows)

    half_test = min(simplerl["test"].num_rows, openmath["test"].num_rows) // 2
    test_rows = list(simplerl["test"])[:half_test] + list(openmath["test"])[:half_test]
    rng.shuffle(test_rows)

    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })


if __name__ == "__main__":
    simplerl = load_from_disk(f"{DATASETS_DIR}/rlvr-simplerl")
    openmath = load_from_disk(f"{DATASETS_DIR}/rlvr-openmath")
    mixed = build_mathmix(simplerl, openmath)
    print(f"MathMix RLVR Dataset - {mixed}")
    print(f"Example row - {mixed['train'][0]}")
    mixed.save_to_disk(f"{DATASETS_DIR}/rlvr-mathmix")
