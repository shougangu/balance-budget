# ABOUTME: GSM8K answer extraction and scoring, matching lm-evaluation-harness.
# ABOUTME: Uses strict (#### regex) then flexible (last-number) extraction with normalization.

import re
from typing import Optional

# lm-eval strict extraction: number right after ####
STRICT_PATTERN = re.compile(r"####\s*(\-?[0-9\.\,]+)")

# lm-eval flexible extraction: last number-like token (including $ amounts)
FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

# lm-eval normalization regexes (applied to both extracted and reference before comparison)
NORMALIZE_REGEXES = [
    re.compile(r","),           # strip commas
    re.compile(r"\$"),          # strip dollar signs
    re.compile(r"(?s).*#### "), # strip everything before "#### "
    re.compile(r"\.$"),         # strip trailing period
]


def extract_gsm8k_answer_strict(response: str) -> Optional[str]:
    """Strict extraction: first match of '#### <number>' pattern."""
    match = STRICT_PATTERN.search(response)
    return match.group(1) if match else None


def extract_gsm8k_answer_flexible(response: str) -> Optional[str]:
    """Flexible extraction: last number-like token in the response."""
    matches = FLEXIBLE_PATTERN.findall(response)
    if not matches:
        return None
    # findall with alternation returns tuples; pick the non-empty group
    last_match = matches[-1]
    return last_match[0] if last_match[0] else last_match[1]


def normalize_answer(answer: str) -> str:
    """Normalize an answer string using lm-eval's regex pipeline."""
    result = answer.lower()
    for pattern in NORMALIZE_REGEXES:
        result = pattern.sub("", result)
    return result.strip()


def is_correct(response: str, reference: str) -> bool:
    """Check if a response matches the reference using lm-eval's approach.

    Tries strict extraction first (#### pattern), falls back to flexible
    (last number). Normalizes both sides before case-insensitive comparison.
    """
    extracted = extract_gsm8k_answer_strict(response)
    if extracted is None:
        extracted = extract_gsm8k_answer_flexible(response)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(reference)
