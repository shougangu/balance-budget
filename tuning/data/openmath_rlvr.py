# ABOUTME: Prompt-only OpenMathInstruct-2 dataset for GRPO/RLVR training.
# ABOUTME: One row per unique math/augmented_math problem with reference_answer for reward function.

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

MATH_SOURCES = {"math", "augmented_math"}


class OpenMathRLVR(HFDataset):
    def __init__(self):
        super().__init__("openmath")

    def _get_rows(self, dataset):
        filtered = dataset.filter(
            lambda rows: [s in MATH_SOURCES for s in rows["problem_source"]],
            batched=True,
        )
        seen = set()
        rows = []
        for row in filtered:
            formatted = COMPMATH_STRING.format(problem=row["problem"])
            if formatted in seen:
                continue
            seen.add(formatted)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": formatted},
                ],
                "reference_answer": row["expected_answer"],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(100, len(rows) - 1), shuffle=False
        )
        print(f"OpenMath RLVR Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    openmath = OpenMathRLVR()
    openmath.load_from_huggingface("nvidia/OpenMathInstruct-2", split="train")
    openmath.format_dataset()
    openmath.clear_old_datasets(prefix="rlvr-openmath")
    openmath.save_dataset_to_disk(save_name="rlvr-openmath")
