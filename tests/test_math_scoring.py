# ABOUTME: Tests for the accept-either math verifier that unions the numeric (lm-eval)
# ABOUTME: and math-verify extraction paths so neither format convention loses credit.

from tuning.evaluation.math_scoring import is_correct


class TestBoxedAnswers:
    """Cases the numeric path alone misreads (docs/EVAL_CALIBRATION.md section 2)."""

    def test_boxed_with_trailing_latex_delimiter(self):
        assert is_correct(r"So the total is $\boxed{3}$.", "3") is True

    def test_boxed_followed_by_prose_number(self):
        assert is_correct(r"He has \boxed{7} dozens of eggs after 4 weeks.", "7") is True

    def test_boxed_with_units(self):
        assert is_correct(r"\boxed{15} gallons", "15") is True

    def test_dollar_math_expression(self):
        assert is_correct(r"$10 - 5.00 = 5.00$", "5") is True


class TestBareNumbers:
    """Cases math-verify alone misreads: a bare number in prose."""

    def test_bare_number_in_prose(self):
        assert is_correct("After adding them up, the answer is 18.", "18") is True

    def test_hash_delimiter(self):
        assert is_correct("Step 1: 9 + 9\n#### 18", "18") is True


class TestWrongAnswers:
    def test_wrong_boxed_and_wrong_prose(self):
        assert is_correct(r"\boxed{7} so 7 apples", "8") is False

    def test_no_number_at_all(self):
        assert is_correct("I do not know.", "8") is False


class TestNonNumericReference:
    """The numeric path compares normalized strings, so it never fires on LaTeX references."""

    def test_fraction_reference_boxed(self):
        assert is_correct(r"\boxed{\frac{3}{4}}", r"\frac{3}{4}") is True

    def test_fraction_reference_bare_prose_is_not_credited(self):
        assert is_correct("The answer is 3/4 so 4.", r"\frac{3}{4}") is False
