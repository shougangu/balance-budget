# ABOUTME: MATH-500 answer scoring via math-verify with #### fallback.
# ABOUTME: Handles both \boxed{} (COMPMATH prompt) and #### (GSM8K training habit) formats.

import re
from math_verify import parse, verify, LatexExtractionConfig

HASH_PATTERN = re.compile(r"####\s*(.*)")
EXTRACTION_CONFIG = [LatexExtractionConfig()]


def is_correct(response: str, reference: str) -> bool:
    """Check if a response matches the reference answer.

    Tries math-verify extraction first (handles \\boxed{} and $...$),
    then falls back to #### extraction for GSM8K-trained models.
    Reference answers are wrapped in \\boxed{} for parsing since
    math-verify can't parse bare numbers/expressions.
    """
    gold = parse(rf"\boxed{{{reference}}}", extraction_config=EXTRACTION_CONFIG)
    if not gold:
        return False

    # Primary: let math-verify extract from \boxed{} and $...$
    pred = parse(response, extraction_config=EXTRACTION_CONFIG)
    if pred and verify(gold, pred):
        return True

    # Fallback: extract from #### pattern (GSM8K format)
    match = HASH_PATTERN.search(response)
    if match:
        extracted = match.group(1).strip()
        pred = parse(rf"\boxed{{{extracted}}}", extraction_config=EXTRACTION_CONFIG)
        if pred and verify(gold, pred):
            return True

    return False
