# ABOUTME: Tests for GSM8K answer extraction and scoring (lm-eval-harness compatible).
# ABOUTME: Covers strict/flexible extraction, normalization, and edge cases.

import pytest
from tuning.evaluation.gsm8k_scoring import (
    extract_gsm8k_answer_strict,
    extract_gsm8k_answer_flexible,
    normalize_answer,
    is_correct,
)


class TestStrictExtraction:
    """Strict extraction: regex '#### (\\-?[0-9\\.\\,]+)' — first match."""

    def test_standard_format(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 42") == "42"

    def test_with_commas(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 1,234") == "1,234"

    def test_negative_number(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### -7") == "-7"

    def test_decimal(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 3.14") == "3.14"

    def test_no_delimiter_returns_none(self):
        assert extract_gsm8k_answer_strict("The answer is 42") is None

    def test_no_number_after_delimiter_returns_none(self):
        assert extract_gsm8k_answer_strict("#### hello") is None

    def test_multiple_delimiters_takes_first(self):
        assert extract_gsm8k_answer_strict("#### 10\n#### 20") == "10"


class TestFlexibleExtraction:
    """Flexible extraction: regex '(-?[$0-9.,]{2,})|(-?[0-9]+)' — last match."""

    def test_number_without_delimiter(self):
        assert extract_gsm8k_answer_flexible("The answer is 42") == "42"

    def test_dollar_amount(self):
        assert extract_gsm8k_answer_flexible("The total cost is $1,234") == "$1,234"

    def test_takes_last_match(self):
        assert extract_gsm8k_answer_flexible("First 10, then 20, finally 30") == "30"

    def test_negative_number(self):
        assert extract_gsm8k_answer_flexible("The result is -5") == "-5"

    def test_no_number_returns_none(self):
        assert extract_gsm8k_answer_flexible("No numbers here") is None

    def test_single_digit(self):
        assert extract_gsm8k_answer_flexible("Answer: 7") == "7"


class TestNormalizeAnswer:
    """Normalization: strip commas, $, '#### ' prefix, trailing period."""

    def test_strip_commas(self):
        assert normalize_answer("1,234,567") == "1234567"

    def test_strip_dollar_sign(self):
        assert normalize_answer("$100") == "100"

    def test_strip_hash_prefix(self):
        assert normalize_answer("some text #### 42") == "42"

    def test_strip_trailing_period(self):
        assert normalize_answer("42.") == "42"

    def test_strip_all_combined(self):
        assert normalize_answer("blah #### $1,000.") == "1000"

    def test_case_insensitive(self):
        assert normalize_answer("ABC") == "abc"

    def test_decimal_not_stripped(self):
        assert normalize_answer("3.14") == "3.14"


class TestIsCorrect:
    """Full pipeline: extract + normalize + compare."""

    def test_strict_match(self):
        assert is_correct("Step 1: blah\n#### 42", "42") is True

    def test_strict_mismatch(self):
        assert is_correct("Step 1: blah\n#### 42", "43") is False

    def test_no_answer_is_incorrect(self):
        assert is_correct("I don't know", "42") is False

    def test_comma_normalization(self):
        assert is_correct("#### 1,000", "1000") is True

    def test_reference_with_hash_prefix(self):
        # GSM8K references often have "#### " prefix
        assert is_correct("#### 42", "#### 42") is True

    def test_flexible_fallback(self):
        # No #### delimiter, but number present — flexible extraction kicks in
        assert is_correct("The answer is 42", "42") is True

    def test_dollar_sign_in_both(self):
        assert is_correct("#### $100", "$100") is True

    def test_trailing_period(self):
        assert is_correct("#### 42.", "42") is True
