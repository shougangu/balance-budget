# ABOUTME: Tests for the standalone OOD benchmark evaluation script (scripts/ood_eval.py).
# ABOUTME: Covers pass@k/sample-count validation and grid cell resolution.

import pytest
from datasets import Dataset

from scripts.ood_eval import MathVerifyEvalStrategy


def _dummy_dataset():
    return Dataset.from_dict({
        "prompt": ["What is 1+1?"],
        "reference_answer": ["2"],
        "messages": [[{"role": "user", "content": "What is 1+1?"}]],
    })


class TestMathVerifyEvalStrategyKValidation:
    def test_rejects_k_greater_than_n_samples(self):
        # pass_at_k returns 1.0 whenever n - c < k, so a k above the sample
        # count silently scores every prompt as solved instead of failing.
        with pytest.raises(ValueError, match="k_values"):
            MathVerifyEvalStrategy(
                "minervamath", _dummy_dataset(), k_values=[1, 2, 4, 8, 16], n_samples=4,
            )

    def test_accepts_k_equal_to_n_samples(self):
        strategy = MathVerifyEvalStrategy(
            "minervamath", _dummy_dataset(), k_values=[1, 2, 4, 8, 16], n_samples=16,
        )
        assert strategy.k_values == [1, 2, 4, 8, 16]
        assert strategy.n_samples == 16

    def test_stopping_k_is_first_k_value(self):
        strategy = MathVerifyEvalStrategy(
            "minervamath", _dummy_dataset(), k_values=[1, 2, 4, 8, 16], n_samples=16,
        )
        assert strategy.stopping_k == 1
