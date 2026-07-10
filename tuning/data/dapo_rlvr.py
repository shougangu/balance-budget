# ABOUTME: Prompt-only DAPO-Math-17k dataset for GRPO/RLVR training.
# ABOUTME: Uses the plain "prompt" field (not source_prompt) with reward_model.ground_truth as reference.

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING


class DAPORLVR(HFDataset):
    def __init__(self):
        super().__init__("dapo")

    def _get_rows(self, dataset):
        seen = set()
        rows = []
        for row in dataset:
            formatted = COMPMATH_STRING.format(problem=row["prompt"])
            if formatted in seen:
                continue
            seen.add(formatted)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": formatted},
                ],
                "reference_answer": row["reward_model"]["ground_truth"],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(100, len(rows) - 1), shuffle=False
        )
        print(f"DAPO-Math RLVR Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    dapo = DAPORLVR()
    dapo.load_from_huggingface("open-r1/DAPO-Math-17k-Processed", "all", split="train")
    dapo.format_dataset()
    dapo.clear_old_datasets(prefix="rlvr-dapo")
    dapo.save_dataset_to_disk(save_name="rlvr-dapo")
