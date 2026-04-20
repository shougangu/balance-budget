# ABOUTME: IF-RLVR dataset for GRPO training with verifiable instruction-following rewards.
# ABOUTME: Loads allenai/IF_multi_constraints_upto5 (95k prompts, up to 5 constraints each).

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING


class IfrlvrRLVR(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="ifrlvr")

    def load_from_huggingface(self, hf_path: str, *args, **kwargs):
        # datasets 3.6.0 doesn't support the 'List' feature type in this dataset's schema,
        # so we load the parquet file directly instead of using load_dataset.
        from huggingface_hub import hf_hub_download
        import pandas as pd

        parquet_path = hf_hub_download(
            hf_path, "data/train-00000-of-00001.parquet", repo_type="dataset"
        )
        df = pd.read_parquet(parquet_path)
        self._dataset = Dataset.from_pandas(df)
        self._raw_dataset = self._dataset
        self.hf_path = hf_path

    def _get_rows(self, dataset):
        seen = set()
        rows = []
        for row in dataset:
            prompt_text = row["messages"][0]["content"]
            if prompt_text in seen:
                continue
            seen.add(prompt_text)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                    {"role": "user", "content": prompt_text},
                ],
                "ground_truth": row["ground_truth"],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(100, len(rows) - 1), shuffle=False
        )
        print(f"IF-RLVR Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    ifrlvr = IfrlvrRLVR()
    ifrlvr.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ifrlvr.format_dataset()
    ifrlvr.clear_old_datasets(prefix="rlvr-ifrlvr")
    ifrlvr.save_dataset_to_disk(save_name="rlvr-ifrlvr")
