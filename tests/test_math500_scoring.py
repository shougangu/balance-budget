# ABOUTME: Unit tests for MATH-500 scoring via math-verify.
# ABOUTME: Validates boxed extraction, #### fallback, symbolic comparison, and edge cases.

from tuning.evaluation.math500_scoring import is_correct


class TestIsCorrectBoxed:
    """Tests where the model uses \\boxed{} format (standard COMPMATH prompt response)."""

    def test_integer_match(self):
        assert is_correct(r"Step 1: ... $\boxed{9}$", "9") is True

    def test_fraction_match(self):
        assert is_correct(r"$\boxed{\frac{14}{3}}$", r"\frac{14}{3}") is True

    def test_equivalent_decimal_fraction(self):
        assert is_correct(r"$\boxed{0.5}$", r"\frac{1}{2}") is True

    def test_wrong_answer(self):
        assert is_correct(r"$\boxed{7}$", "9") is False

    def test_negative_number(self):
        assert is_correct(r"$\boxed{-3}$", "-3") is True

    def test_sqrt_expression(self):
        assert is_correct(r"$\boxed{11\sqrt{2}}$", r"11\sqrt{2}") is True


class TestIsCorrectHashFallback:
    """Tests where the model uses #### format (GSM8K training habit)."""

    def test_hash_integer(self):
        assert is_correct("Step 1: ...\n#### 9", "9") is True

    def test_hash_wrong_answer(self):
        assert is_correct("Step 1: ...\n#### 7", "9") is False

    def test_hash_with_whitespace(self):
        assert is_correct("#### 42 ", "42") is True


class TestIsCorrectNoAnswer:
    """Tests where no answer can be extracted."""

    def test_plain_text_no_extraction(self):
        assert is_correct("The answer is 9", "9") is False

    def test_empty_response(self):
        assert is_correct("", "9") is False


def test_get_math500_test_dataset_loads():
    """get_math500_test_dataset returns a Dataset with expected columns and COMPMATH prompt."""
    from unittest.mock import patch
    from datasets import Dataset

    mock_hf_dataset = Dataset.from_dict({
        "problem": ["What is 2+2?", "Solve x^2=4"],
        "answer": ["4", "2"],
        "solution": ["2+2=4", "x=2"],
        "subject": ["Algebra", "Algebra"],
        "level": [1, 2],
        "unique_id": ["test/1", "test/2"],
    })

    with patch("datasets.load_dataset", return_value=mock_hf_dataset):
        from tuning.data.test_dataset import get_math500_test_dataset
        ds = get_math500_test_dataset(num_prompts=2)
        assert "messages" in ds.column_names
        assert "prompt" in ds.column_names
        assert "reference_answer" in ds.column_names
        assert len(ds) == 2
        # System message uses COMPMATH (asks for \boxed{} format)
        from tuning.data.config import SYSTEM_MESSAGE_COMPMATH
        assert ds[0]["messages"][0]["content"] == SYSTEM_MESSAGE_COMPMATH
        # User message uses COMPMATH_STRING template
        assert "Problem:" in ds[0]["messages"][1]["content"]
        assert "What is 2+2?" in ds[0]["messages"][1]["content"]
