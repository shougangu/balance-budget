# ABOUTME: MATH-500 answer scoring via math-verify with #### fallback.
# ABOUTME: Handles both \boxed{} (COMPMATH prompt) and #### (GSM8K training habit) formats.

import re
from math_verify import parse, verify, LatexExtractionConfig
from math_verify.errors import TimeoutException

HASH_PATTERN = re.compile(r"####\s*(.*)")
EXTRACTION_CONFIG = [LatexExtractionConfig()]
# math-verify's own default for parse/verify timeouts.
DEFAULT_TIMEOUT_SECONDS = 3


def is_correct(response: str, reference: str, timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Check if a response matches the reference answer.

    Tries math-verify extraction first (handles \\boxed{} and $...$),
    then falls back to #### extraction for GSM8K-trained models.
    Reference answers are wrapped in \\boxed{} for parsing since
    math-verify can't parse bare numbers/expressions.

    timeout_seconds bounds each parse/verify through math-verify's
    signal.alarm; pass None off the main thread, where signals are unavailable.
    """
    try:
        return _is_correct_inner(response, reference, timeout_seconds)
    except TimeoutException:
        print(f"[math500_scoring] TimeoutException: ref={reference!r}, response={response[:200]!r}")
        return False


def _is_correct_inner(response: str, reference: str, timeout_seconds: int | None) -> bool:
    def parse_expr(text):
        return parse(text, extraction_config=EXTRACTION_CONFIG, parsing_timeout=timeout_seconds)

    gold = parse_expr(rf"\boxed{{{reference}}}")
    if not gold:
        return False

    # Primary: let math-verify extract from \boxed{} and $...$
    pred = parse_expr(response)
    if pred and verify(gold, pred, timeout_seconds=timeout_seconds):
        return True

    # Fallback: extract from #### pattern (GSM8K format)
    match = HASH_PATTERN.search(response)
    if match:
        extracted = match.group(1).strip()
        pred = parse_expr(rf"\boxed{{{extracted}}}")
        if pred and verify(gold, pred, timeout_seconds=timeout_seconds):
            return True

    return False
