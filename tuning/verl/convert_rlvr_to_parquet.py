# ABOUTME: Converts an on-disk rlvr-* dataset into verl's parquet schema.
# ABOUTME: Prompts stay byte-identical to the SFT-side rendering; answers become ground_truth.

import argparse
import os

import pyarrow.parquet
from datasets import load_from_disk

from tuning.config import DATASETS_DIR


def to_verl_row(row: dict, index: int, data_source: str, split: str) -> dict:
    return {
        "data_source": data_source,
        "prompt": row["prompt"],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["reference_answer"]},
        "extra_info": {"index": index, "split": split},
    }


def convert(dataset_name: str, out_dir: str) -> dict:
    """Write train/test parquet files for one rlvr dataset; returns the paths."""
    stub = f"rlvr-{dataset_name}"
    dataset = load_from_disk(os.path.join(DATASETS_DIR, stub))
    data_source = f"balance-budget/{stub}"
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split in dataset:
        converted = dataset[split].map(
            lambda row, index: to_verl_row(row, index, data_source, split),
            with_indices=True,
            remove_columns=dataset[split].column_names,
        )
        paths[split] = os.path.join(out_dir, f"{stub}.{split}.parquet")
        # datasets.to_parquet passes writer kwargs this venv's pyarrow lacks;
        # write the arrow table directly.
        pyarrow.parquet.write_table(converted.data.table, paths[split])
        print(f"[convert] {split}: {converted.num_rows} rows -> {paths[split]}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rlvr-* datasets to verl parquet.")
    parser.add_argument("--dataset", required=True, help="e.g. dapo (reads rlvr-dapo)")
    parser.add_argument("--out-dir", default=os.path.join(DATASETS_DIR, "verl"))
    args = parser.parse_args()
    convert(args.dataset, args.out_dir)
