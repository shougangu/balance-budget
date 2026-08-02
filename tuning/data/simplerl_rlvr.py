# ABOUTME: SimpleRL-Zoo dataset loader for GRPO/RLVR training with three difficulty tiers.
# ABOUTME: Downloads pre-split parquet from hkust-nlp/SimpleRL-Zoo-Data, applies OpenMath chat formatting.

import random

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
import pandas as pd

from tuning.config import DATASETS_DIR
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

SIMPLERL_TIERS = {
    "easy":   "simplelr_abel_gsm8k_level1",
    "medium": "simplelr_abel_level1to4",
    "hard":   "simplelr_abel_level3to5",
}


def combine_tiers(tier_datasets, seed=42):
    """Merge per-tier formatted datasets into one, dropping prompts already seen in an
    earlier tier so the combined dataset has no duplicate prompts. Each split is shuffled
    so the tiers are interleaved instead of concatenated in tier order; the test split is
    then capped at 100 rows, matching each tier's format_dataset()."""
    rng = random.Random(seed)
    splits = {}
    for split in ("train", "test"):
        seen = set()
        rows = []
        for dataset in tier_datasets:
            for row in dataset[split]:
                key = row["prompt"][-1]["content"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        rng.shuffle(rows)
        if split == "test":
            rows = rows[:100]
        splits[split] = Dataset.from_list(rows)
    return DatasetDict(splits)


class SimpleRLRLVR(HFDataset):
    def __init__(self, difficulty: str):
        if difficulty not in SIMPLERL_TIERS:
            raise ValueError(f"difficulty must be one of {set(SIMPLERL_TIERS)}, got {difficulty!r}")
        self._difficulty = difficulty
        self._subset = SIMPLERL_TIERS[difficulty]
        super().__init__(dataset_name=f"simplerl-{difficulty}")

    def load_from_huggingface(self, hf_path: str, *args, **kwargs):
        self.hf_path = hf_path
        train_path = hf_hub_download(
            hf_path, f"{self._subset}/train.parquet", repo_type="dataset"
        )
        test_path = hf_hub_download(
            hf_path, f"{self._subset}/test.parquet", repo_type="dataset"
        )
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        self._dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        })
        self._raw_dataset = self._dataset

    def _get_rows(self, dataset):
        seen = set()
        rows = []
        for row in dataset:
            question = row["extra_info"]["question"]
            ground_truth = row["reward_model"]["ground_truth"]
            formatted = COMPMATH_STRING.format(problem=question)
            if formatted in seen:
                continue
            seen.add(formatted)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": formatted},
                ],
                "reference_answer": ground_truth,
            })
        return rows

    def format_dataset(self):
        train_rows = self._get_rows(self._dataset["train"])
        test_rows = self._get_rows(self._dataset["test"])[:200]
        self._dataset = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        })
        print(f"SimpleRL-Zoo ({self._difficulty}) Dataset - {self._dataset}")
        print(f"Example row - {self._dataset['train'][0]}")


if __name__ == "__main__":
    tier_datasets = []
    for difficulty in SIMPLERL_TIERS:
        ds = SimpleRLRLVR(difficulty)
        ds.load_from_huggingface("hkust-nlp/SimpleRL-Zoo-Data")
        ds.format_dataset()
        tier_datasets.append(ds.get_dataset())
        save_name = f"rlvr-simplerl-{difficulty}"
        ds.clear_old_datasets(prefix=save_name)
        ds.save_dataset_to_disk(save_name=save_name)

    combined = combine_tiers(tier_datasets)
    print(f"SimpleRL-Zoo (combined) Dataset - {combined}")
    print(f"Example row - {combined['train'][0]}")
    combined.save_to_disk(f"{DATASETS_DIR}/rlvr-simplerl")
