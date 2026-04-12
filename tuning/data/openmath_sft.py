# ABOUTME: SFT dataset from OpenMathInstruct-2 for complex math training.
# ABOUTME: Filters to math/augmented_math sources, keeps all (problem, solution) pairs.

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

MATH_SOURCES = {"math", "augmented_math"}


class OpenMathSFT(HFDataset):
    def __init__(self):
        super().__init__("openmath")

    def _get_rows(self, dataset):
        rows = []
        for i in range(len(dataset)):
            row = dataset[i]
            if row["problem_source"] not in MATH_SOURCES:
                continue
            prompt = COMPMATH_STRING.format(problem=row["problem"])
            rows.append({
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["generated_solution"]},
                ],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(200, len(rows) - 1), shuffle=False
        )
        print(f"OpenMath SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    openmath = OpenMathSFT()
    openmath.load_from_huggingface("nvidia/OpenMathInstruct-2", split="train")
    openmath.format_dataset()
    openmath.clear_old_datasets(prefix="sft-openmath")
    openmath.save_dataset_to_disk(save_name="sft-openmath")
