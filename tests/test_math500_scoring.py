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
