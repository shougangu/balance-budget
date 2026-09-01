# ABOUTME: Accept-either math verifier: a response is correct if the numeric (lm-eval)
# ABOUTME: path or the math-verify path recovers the reference answer.

from tuning.evaluation.gsm8k_scoring import is_correct as numeric_is_correct
from tuning.evaluation.math500_scoring import DEFAULT_TIMEOUT_SECONDS
from tuning.evaluation.math500_scoring import is_correct as math_verify_is_correct


def is_correct(response: str, reference: str,
               timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Return True if either extraction path matches the reference.

    The numeric path (#### then last number, lm-eval semantics) runs first
    because it is far cheaper than math-verify and its result is unaffected
    by order. It compares normalized strings, so it cannot fire on a LaTeX
    reference such as \\frac{3}{4}; those are decided by math-verify alone.
    """
    return (numeric_is_correct(response, reference)
            or math_verify_is_correct(response, reference, timeout_seconds))
