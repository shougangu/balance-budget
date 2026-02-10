from datasets import DatasetDict
import random

random.seed(42)


def get_random_train_subset(dataset: DatasetDict, train_size: int) -> DatasetDict:

    train_split = dataset["train"]
    test_split = dataset["test"]

    # Cap train_size at available dataset size
    actual_train_size = min(train_size, len(train_split))
    
    if actual_train_size < train_size:
        print(f"[WARNING] Requested train_size={train_size} exceeds dataset size={len(train_split)}")
        print(f"[WARNING] Using full dataset with {actual_train_size} examples instead")

    random_subset = random.sample(range(len(train_split)), actual_train_size)
    train_split = train_split.select(random_subset)

    sampled_dataset = DatasetDict()

    sampled_dataset["train"] = train_split
    sampled_dataset["test"] = test_split

    return sampled_dataset
