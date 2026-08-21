# ABOUTME: SFT dataset from OpenMathInstruct-2 for complex math training.
# ABOUTME: Keeps every MATH and GSM8K source, all (problem, solution) pairs.

import argparse

import numpy as np

from datasets import Dataset, DatasetDict, load_dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING
from tuning.data.heldout_math_eval import build_heldout_math_eval

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

    def format_dataset(self, length_percentile=None, eval_dataset=None):
        # Every problem in this corpus carries many generated solutions, so slicing rows
        # off the end would hold out solutions whose problems remain in training. The
        # evaluation split is sourced from problems the corpus was never seeded from.
        formatted = self._format_rows(self._dataset, length_percentile=length_percentile)
        if eval_dataset is None:
            eval_dataset = build_heldout_math_eval()
        formatted_dataset = DatasetDict({"train": formatted, "test": eval_dataset})
        print(f"OpenMath SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


CORPUS = "nvidia/OpenMathInstruct-2"


def build_openmath_sft(split="train", save_name="sft-openmath", clear_existing=False,
                       length_percentile=None):
    """Build and save an OpenMath SFT dataset from one split of the corpus.

    The corpus ships fair-downsampled splits (train_1M, train_2M, train_5M) whose
    solutions are spread across questions instead of concentrated on the ones that
    were easiest to generate for. Clearing is opt-in and scoped to save_name so a
    rebuild cannot take sibling datasets with it.
    """
    openmath = OpenMathSFT()
    openmath._dataset = load_dataset(CORPUS, split=split)
    openmath._raw_dataset = openmath._dataset
    openmath.hf_path = CORPUS
    if clear_existing:
        openmath.clear_old_datasets(prefix=save_name)
    openmath.format_dataset(length_percentile=length_percentile)
    openmath.save_dataset_to_disk(save_name=save_name)
    return openmath.get_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an OpenMath SFT dataset.")
    parser.add_argument("--split", default="train",
                        help="Corpus split, e.g. train / train_1M / train_2M / train_5M")
    parser.add_argument("--save-name", default="sft-openmath")
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--length-percentile", type=float, default=None)
    args = parser.parse_args()
    build_openmath_sft(
        split=args.split,
        save_name=args.save_name,
        clear_existing=args.clear_existing,
        length_percentile=args.length_percentile,
    )

    # openmath_lenp95 = OpenMathSFT()
    # openmath_lenp95.load_from_huggingface("nvidia/OpenMathInstruct-2", split="train")
    # openmath_lenp95.format_dataset(length_percentile=95)
    # openmath_lenp95.save_dataset_to_disk(save_name="sft-openmath-lenp95")
