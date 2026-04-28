from datasets import DatasetDict
import random


def get_random_train_subset(dataset: DatasetDict, train_size: int,
                            unique_column: str = None) -> DatasetDict:

    train_split = dataset["train"]
    test_split = dataset["test"]

    # Cap train_size at available dataset size
    actual_train_size = min(train_size, len(train_split))

    if actual_train_size < train_size:
        print(f"[WARNING] Requested train_size={train_size} exceeds full dataset size={len(train_split)}")
        print(f"[WARNING] Using full dataset with {actual_train_size} examples instead")

    if unique_column:
        # Take one row per unique value first, then fill with duplicates
        seen = set()
        unique_indices = []
        duplicate_indices = []
        for i in range(len(train_split)):
            val = train_split[i][unique_column]
            try:
                key = val
                hash(key)
            except TypeError:
                key = repr(val)  # chat-format prompts (list of dicts) etc.
            if key not in seen:
                seen.add(key)
                unique_indices.append(i)
            else:
                duplicate_indices.append(i)

        if len(unique_indices) > actual_train_size:
            random_subset = sorted(random.sample(unique_indices, actual_train_size))
        else:
            remaining = actual_train_size - len(unique_indices)
            fill = random.sample(duplicate_indices, min(remaining, len(duplicate_indices)))
            random_subset = sorted(unique_indices + fill)
    else:
        random_subset = random.sample(range(len(train_split)), actual_train_size)

    train_split = train_split.select(random_subset)

    sampled_dataset = DatasetDict()

    sampled_dataset["train"] = train_split
    sampled_dataset["test"] = test_split

    return sampled_dataset
