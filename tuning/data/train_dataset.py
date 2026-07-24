from typing import Union
from tuning.training.config_training import DatasetConfig, PTRunConfig, SFTRunConfig
from tuning.data.utils import get_random_train_subset
from datasets import Dataset, load_from_disk, DatasetDict
from tuning.config import DATASETS_DIR
from pathlib import Path

import torch.distributed as dist

def get_train_dataset(run_config: Union[PTRunConfig, SFTRunConfig]) -> DatasetDict:

    print(f"Getting train dataset for run config: {run_config}")

    if run_config.run_type == "pt":

        # if run_config.sft_run_config is None:
        #     dataset_stub = f"pt-{run_config.dataset_config.dataset}"
        # else:
        #     sft_run_name = run_config.sft_run_config.run_name
        #     sft_run_name = sft_run_name.replace("_", "|").replace("-", "|")
        #     dataset_stub = f"pt-{sft_run_name}"

        if run_config.dataset_config.dataset_type == "rlvr":
            dataset_stub = f"rlvr-{run_config.dataset_config.dataset}"
        else:
            dataset_stub = f"pt-{run_config.dataset_config.dataset}"
    elif run_config.run_type == "sft":
        dataset_stub = f"sft-{run_config.dataset_config.dataset}"
    
    save_name = f"{dataset_stub}-{run_config.dataset_config.train_size}"
    check_path = f"{DATASETS_DIR}/{save_name}"

    print(f"Checking for dataset at {check_path}")
    if Path(check_path).exists():
        print(f"Dataset already exists at {check_path}")

        try:
            full_dataset = load_from_disk(check_path)
        except (IndexError, FileNotFoundError):
            # datasets 4.8.2 can't reload 0-row splits (no arrow file written for 0/0 shards)
            test = Dataset.load_from_disk(f"{check_path}/test")
            train = Dataset.from_dict({col: [] for col in test.column_names})
            full_dataset = DatasetDict({"train": train, "test": test})
        print(f"Sampled dataset: {full_dataset}")
        if len(full_dataset['train']) > 0:
            print(f"Example training row: {full_dataset['train'][0]}")
        print(f"Example evaluation row: {full_dataset['test'][0]}")

        return full_dataset

    # Cache miss: under DDP, only rank 0 builds and writes; other ranks wait at
    # the barrier and then load_from_disk the freshly written cache. Concurrent
    # save_to_disk on the same path corrupts arrow shards silently.
    is_dist = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if is_dist else 0

    train_size = run_config.dataset_config.train_size
    full_dataset_path = f"{DATASETS_DIR}/{dataset_stub}"
    full_dataset = load_from_disk(full_dataset_path)

    # A request that covers the entire split does not need a sampled cache.
    # Returning the memory-mapped base dataset avoids materializing a redundant
    # random permutation of every row. All DDP ranks can safely read it.
    if train_size >= len(full_dataset["train"]):
        if rank == 0:
            if train_size > len(full_dataset["train"]):
                print(
                    f"[WARNING] Requested train_size={train_size} exceeds full "
                    f"dataset size={len(full_dataset['train'])}"
                )
            print(
                f"Using full dataset directly from {full_dataset_path}; "
                "no sampled cache will be written"
            )
        return full_dataset

    sampled_dataset = None
    if rank == 0:
        print(f"Loading dataset from {full_dataset_path}")
        print(f"Full dataset: {full_dataset}")

        unique_column = "prompt" if "prompt" in full_dataset["train"].column_names else None
        sampled_dataset = get_random_train_subset(full_dataset, train_size, unique_column=None)
        print(f"Sampled dataset: {sampled_dataset}")
        if len(sampled_dataset['train']) > 0:
            print(f"Example training row: {sampled_dataset['train'][0]}")
        print(f"Example evaluation row: {sampled_dataset['test'][0]}")

        sampled_dataset.save_to_disk(check_path)

    if is_dist:
        dist.barrier()

    if rank != 0:
        sampled_dataset = load_from_disk(check_path)

    return sampled_dataset

if __name__ == "__main__":

    run_config = SFTRunConfig(
            model_name="llama3-8B",
            task_name="math",
            dataset_config=DatasetConfig(
                dataset="gsm8k", #gsm8k
                dataset_type="sft",
                train_size=100,
            ),
            do_training=True,
        )

    dataset = get_train_dataset(run_config=run_config)

    print(f"Dataset: {dataset}")
