# ABOUTME: Tests for the held-out math evaluation split built from MATH-500 and GSM8K test.
# ABOUTME: Covers answer-format normalisation and the openmath prompt rendering.

from tuning.data.config import COMPMATH_STRING, SYSTEM_MESSAGE_OPENMATH
from tuning.data.heldout_math_eval import (
    boxed_gsm8k_solution,
    build_heldout_math_eval,
    to_openmath_row,
)


def test_gsm8k_marker_becomes_boxed():
    """GSM8K's '#### 42' tail is rewritten into the boxed form the system prompt asks for."""
    solution = "Half of 84 is 42.\n#### 42"
    assert boxed_gsm8k_solution(solution) == "Half of 84 is 42.\n$\\boxed{42}$"


def test_gsm8k_reasoning_is_preserved():
    """Only the final-answer marker changes; the reasoning above it survives intact."""
    solution = "Step one.\nStep two.\n#### 7"
    result = boxed_gsm8k_solution(solution)
    assert "Step one." in result
    assert "Step two." in result
    assert "####" not in result


def test_solution_without_marker_is_unchanged():
    """A solution that already lacks the marker passes through untouched."""
    solution = "The answer is $\\boxed{5}$"
    assert boxed_gsm8k_solution(solution) == solution


def test_row_uses_openmath_prompt_format():
    """Rows render with the same prompt and system message as the openmath SFT builder."""
    row = to_openmath_row("What is 2+2?", "It is 4. $\\boxed{4}$")

    assert row["prompt"] == COMPMATH_STRING.format(problem="What is 2+2?")
    assert row["messages"][0] == {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH}
    assert row["messages"][1] == {"role": "user", "content": row["prompt"]}
    assert row["messages"][2]["role"] == "assistant"


def test_row_columns_match_openmath_sft():
    """Column set matches sft-openmath so the split is a drop-in eval set."""
    row = to_openmath_row("q", "a")
    assert set(row) == {"prompt", "messages"}


def test_build_combines_both_sources():
    """The built split is MATH-500 (500) plus the GSM8K test set (1319)."""
    dataset = build_heldout_math_eval()

    assert dataset.column_names == ["prompt", "messages"]
    assert dataset.num_rows == 1819
    assert len(set(dataset["prompt"])) == 1819


def test_build_leaves_no_gsm8k_markers():
    """No row keeps GSM8K's '####' answer marker after normalisation."""
    dataset = build_heldout_math_eval()
    assert not [r for r in dataset["messages"] if "####" in r[-1]["content"]]
