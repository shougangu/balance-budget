# ABOUTME: SFT dataset from OpenMathInstruct-2 for complex math training.
# ABOUTME: Filters to math/augmented_math sources, keeps all (problem, solution) pairs.

import numpy as np

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

MATH_SOURCES = {"math", "augmented_math", "augmented_gsm8k", "gsm8k"}


class OpenMathSFT(HFDataset):
    def __init__(self):
        super().__init__("openmath")

    def _format_rows(self, dataset, length_percentile=None):
        filtered = dataset.filter(
            lambda rows: [s in MATH_SOURCES for s in rows["problem_source"]],
            batched=True,
        )

        if length_percentile is not None:
            lengths = [len(s) for s in filtered["generated_solution"]]
            threshold = float(np.percentile(lengths, length_percentile))
            print(
                f"Filtering to responses with character length >= {threshold:.0f} "
                f"(p{length_percentile} of {len(lengths)} rows)"
            )
            filtered = filtered.filter(
                lambda row: len(row["generated_solution"]) >= threshold
            )

        def transform(row):
            prompt = COMPMATH_STRING.format(problem=row["problem"])
            return {
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["generated_solution"]},
                ],
            }

        return filtered.map(
            transform,
            remove_columns=filtered.column_names,
        )

    def format_dataset(self, length_percentile=None):
        formatted = self._format_rows(self._dataset, length_percentile=length_percentile)
        formatted_dataset = formatted.train_test_split(
            test_size=min(200, len(formatted) - 1), shuffle=False
        )
        print(f"OpenMath SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    openmath = OpenMathSFT()
    openmath.load_from_huggingface("nvidia/OpenMathInstruct-2", split="train")
    openmath.clear_old_datasets(prefix="sft-openmath")
    openmath.format_dataset()
    openmath.save_dataset_to_disk(save_name="sft-openmath")

    # openmath_lenp95 = OpenMathSFT()
    # openmath_lenp95.load_from_huggingface("nvidia/OpenMathInstruct-2", split="train")
    # openmath_lenp95.format_dataset(length_percentile=95)
    # openmath_lenp95.save_dataset_to_disk(save_name="sft-openmath-lenp95")
