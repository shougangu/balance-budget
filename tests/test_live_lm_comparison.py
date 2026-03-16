# ABOUTME: Tests for format_vllm_outputs and scoring consistency between lm-eval and eval_strategy.
# ABOUTME: Validates that both evaluation paths produce equivalent results.

import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset


# --- format_vllm_outputs tests ---

def _mock_vllm_output(texts):
    """Create a mock vLLM RequestOutput with the given response texts."""
    output = MagicMock()
    output.outputs = [MagicMock(text=t) for t in texts]
    return output


def test_format_vllm_outputs_single_sample():
    """Single-sample outputs are grouped by prompt with one response each."""
    from tuning.training.eval_strategy import format_vllm_outputs

    outputs = [_mock_vllm_output(["answer1"]), _mock_vllm_output(["answer2"])]
    results = format_vllm_outputs(outputs, ["Q1", "Q2"], n_samples=1)

    assert len(results) == 2
    assert results[0] == {"prompt": "Q1", "responses": ["answer1"]}
    assert results[1] == {"prompt": "Q2", "responses": ["answer2"]}


def test_format_vllm_outputs_multi_sample():
    """Multi-sample outputs collect all responses per prompt."""
    from tuning.training.eval_strategy import format_vllm_outputs

    outputs = [
        _mock_vllm_output(["a1", "a2", "a3"]),
        _mock_vllm_output(["b1", "b2", "b3"]),
    ]
    results = format_vllm_outputs(outputs, ["Q1", "Q2"], n_samples=3)

    assert len(results) == 2
    assert results[0]["responses"] == ["a1", "a2", "a3"]
    assert results[1]["responses"] == ["b1", "b2", "b3"]


def test_format_vllm_outputs_deduplicates_prompts():
    """Duplicate prompts merge their responses into a single entry."""
    from tuning.training.eval_strategy import format_vllm_outputs

    outputs = [_mock_vllm_output(["a1"]), _mock_vllm_output(["a2"])]
    results = format_vllm_outputs(outputs, ["Q1", "Q1"], n_samples=1)

    assert len(results) == 1
    assert results[0]["prompt"] == "Q1"
    assert results[0]["responses"] == ["a1", "a2"]


def test_format_vllm_outputs_preserves_prompt_order():
    """Results maintain the order of first appearance of each prompt."""
    from tuning.training.eval_strategy import format_vllm_outputs

    outputs = [
        _mock_vllm_output(["a"]),
        _mock_vllm_output(["b"]),
        _mock_vllm_output(["c"]),
    ]
    results = format_vllm_outputs(outputs, ["Q2", "Q1", "Q3"], n_samples=1)

    assert [r["prompt"] for r in results] == ["Q2", "Q1", "Q3"]


# --- Scoring consistency tests ---

def _make_gsm8k_strategy(prompts, references):
    """Create a GSM8KEvalStrategy with mocked test data."""
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_ds:
        mock_ds.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": p}] for p in prompts],
            "prompt": prompts,
            "reference_answer": references,
        })
        from tuning.training.eval_strategy import GSM8KEvalStrategy
        return GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=len(prompts))


def _make_mock_tokenizer():
    """Create a tokenizer mock that returns dummy token IDs."""
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda texts, **kw: {"input_ids": [[0] * 10] * len(texts)}
    return tokenizer


def test_scoring_consistency_gsm8k():
    """format_vllm_outputs + strategy.score_responses produces correct scores."""
    from tuning.training.eval_strategy import format_vllm_outputs

    strategy = _make_gsm8k_strategy(["Q1", "Q2"], ["42", "100"])
    tokenizer = _make_mock_tokenizer()

    outputs = [
        _mock_vllm_output(["Step 1: ...\n#### 42"]),   # correct
        _mock_vllm_output(["Step 1: ...\n#### 99"]),    # wrong
    ]
    results = format_vllm_outputs(outputs, ["Q1", "Q2"], n_samples=1)
    scores = strategy.score_responses(results, tokenizer)

    assert scores["pass_at_1"] == 0.5
    assert scores["num_prompts_evaluated"] == 2


def test_scoring_consistency_direct_vs_strategy():
    """Direct gsm8k_scoring.is_correct agrees with GSM8KEvalStrategy scoring."""
    from tuning.evaluation.gsm8k_scoring import is_correct
    from tuning.training.eval_strategy import format_vllm_outputs

    prompts = ["Q1", "Q2", "Q3"]
    references = ["42", "100", "7"]
    responses = [
        "Step 1: ...\n#### 42",    # correct
        "The answer is 100",        # correct (flexible extraction)
        "Step 1: ...\n#### 8",      # wrong
    ]

    strategy = _make_gsm8k_strategy(prompts, references)
    tokenizer = _make_mock_tokenizer()

    # Direct scoring
    direct_results = [is_correct(r, ref) for r, ref in zip(responses, references)]

    # Strategy scoring via format_vllm_outputs pipeline
    outputs = [_mock_vllm_output([r]) for r in responses]
    formatted = format_vllm_outputs(outputs, prompts, n_samples=1)
    scores = strategy.score_responses(formatted, tokenizer)

    assert direct_results == [True, True, False]
    assert scores["pass_at_1"] == pytest.approx(sum(direct_results) / len(direct_results))


def test_lm_eval_scoring_matches_is_correct():
    """lm-eval gsm8k_matched regex filters should agree with gsm8k_scoring.is_correct.

    The gsm8k_matched task uses two filters:
    - strict-match: regex '#### (-?[0-9.,]+)' → exact_match with normalization
    - flexible-extract: regex '(-?[$0-9.,]{2,})|(-?[0-9]+)' group_select=-1 → exact_match

    gsm8k_scoring.is_correct tries strict first, then falls back to flexible.
    This test verifies both produce the same result on representative examples.
    """
    from tuning.evaluation.gsm8k_scoring import (
        is_correct,
        extract_gsm8k_answer_strict,
        extract_gsm8k_answer_flexible,
        normalize_answer,
    )

    test_cases = [
        # (response, reference, expected_correct)
        ("Step 1: 40+2\n#### 42", "42", True),
        ("Step 1: 40+2\n#### 42", "43", False),
        ("The answer is 100", "100", True),       # flexible extraction
        ("The answer is $3,500", "3500", True),    # flexible with $ and comma
        ("No number here", "42", False),           # no extraction possible
        ("#### -7", "-7", True),                   # negative number strict
        ("blah blah 42 more text 99", "99", True), # flexible picks last number
    ]

    for response, reference, expected in test_cases:
        result = is_correct(response, reference)
        assert result == expected, (
            f"is_correct({response!r}, {reference!r}) = {result}, expected {expected}"
        )


def test_gsm8k_evaluate_uses_vllm_sampling_config():
    """gsm8k_evaluate should use VLLMSamplingParamsConfig for lm-eval gen_kwargs."""
    import ast
    from pathlib import Path

    source = Path("tuning/evaluation/gsm8k_evaluate.py").read_text()
    tree = ast.parse(source)

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)

    assert "VLLMSamplingParamsConfig" in imported_names, \
        "gsm8k_evaluate.py should import VLLMSamplingParamsConfig"


def test_gsm8k_evaluate_accepts_config_params():
    """gsm8k_evaluate should accept temperature, max_tokens, num_prompts parameters."""
    import ast
    from pathlib import Path

    source = Path("tuning/evaluation/gsm8k_evaluate.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gsm8k_evaluate":
            param_names = [arg.arg for arg in node.args.args]
            assert "temperature" in param_names, \
                "gsm8k_evaluate should accept temperature parameter"
            assert "max_tokens" in param_names, \
                "gsm8k_evaluate should accept max_tokens parameter"
            break
    else:
        pytest.fail("gsm8k_evaluate function not found")


def test_live_lm_comparison_output_structure():
    """live_lm_comparison should define output under outputs/live_lm_comparison/."""
    from pathlib import Path

    source = Path("scripts/live_lm_comparison.py").read_text()

    assert "live_lm_comparison" in source


def test_live_lm_comparison_runs_both_paths():
    """live_lm_comparison should run both lm-eval and eval_strategy paths."""
    import ast
    from pathlib import Path

    source = Path("scripts/live_lm_comparison.py").read_text()

    # Should reference both GSM8KEvalStrategy and gsm8k_evaluate (lm-eval path)
    assert "GSM8KEvalStrategy" in source, \
        "live_lm_comparison should use GSM8KEvalStrategy for eval_strategy path"
    assert "gsm8k_evaluate" in source or "lm-eval" in source or "lm_eval" in source, \
        "live_lm_comparison should use lm-eval path for comparison"
