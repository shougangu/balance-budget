# ABOUTME: Tests for eval strategy ABC, IFEval, and GSM8K implementations.
# ABOUTME: Validates interface conformance and pass@k scoring logic.

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset


def test_ifeval_strategy_implements_interface():
    """IFEvalStrategy must implement all EvalStrategy abstract methods."""
    from tuning.training.eval_strategy import IFEvalStrategy, EvalStrategy
    assert issubclass(IFEvalStrategy, EvalStrategy)

    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1], n_samples=1, num_prompts=1, strict=True)
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_pass_at_k_computation():
    """pass_at_k(n, c, k) should compute correct probabilities."""
    from tuning.training.eval_strategy import pass_at_k
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 0, 1) == 0.0
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-6


def test_ifeval_stopping_metric():
    """stopping_metric should return the key for the first k value."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1, 5], n_samples=5, num_prompts=1, strict=True)
        assert strategy.stopping_metric() == "pass_at_1"


def test_ifeval_label_prefix():
    """label_prefix should return 'p@{stopping_k}'."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1], n_samples=1, num_prompts=1, strict=True)
        assert strategy.label_prefix == "p@1"


def test_ifeval_wandb_metrics():
    """wandb_metrics should prefix scores with eval/."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1], n_samples=1, num_prompts=1, strict=True)
        scores = {"pass_at_1": 0.72, "avg_response_length_tokens": 150.0, "num_prompts_evaluated": 10}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/pass_at_1" in wandb_dict
        assert "eval/avg_response_length_tokens" in wandb_dict


def test_callback_accepts_eval_strategy():
    """PassAtKStoppingCallback should accept primary_eval and monitor_evals."""
    from tuning.training.passk_callback import PassAtKStoppingCallback
    import inspect
    sig = inspect.signature(PassAtKStoppingCallback.__init__)
    params = list(sig.parameters.keys())
    assert "primary_eval" in params
    assert "monitor_evals" in params


def test_passk_config_no_eval_fields():
    """PassAtKConfig should not have eval-specific fields (those belong on strategy constructors)."""
    from tuning.training.config_training import PassAtKConfig
    config = PassAtKConfig()
    assert not hasattr(config, "strict"), "strict belongs on eval strategy constructor"
    assert not hasattr(config, "k_values"), "k_values belongs on eval strategy constructor"
    assert not hasattr(config, "n_samples"), "n_samples belongs on eval strategy constructor"
    assert not hasattr(config, "num_prompts"), "num_prompts belongs on eval strategy constructor"


def test_ifeval_strategy_accepts_direct_params():
    """IFEvalStrategy should accept params directly (no config object)."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1, 5], n_samples=5, num_prompts=1, strict=True)
        assert strategy.k_values == [1, 5]
        assert strategy.n_samples == 5
        assert strategy.strict is True
        assert strategy.stopping_metric() == "pass_at_1"


def test_gsm8k_strategy_implements_interface():
    """GSM8KEvalStrategy must implement all EvalStrategy abstract methods."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy, EvalStrategy
    assert issubclass(GSM8KEvalStrategy, EvalStrategy)
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_gsm8k_stopping_metric():
    """GSM8K stopping metric should follow pass@k convention: 'pass_at_{k}'."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1, 5], n_samples=5, num_prompts=1)
        assert strategy.stopping_metric() == "pass_at_1"


def test_gsm8k_label_prefix():
    """GSM8K label_prefix should be 'gsm8k-p@{stopping_k}'."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.label_prefix == "gsm8k-p@1"


def test_gsm8k_score_responses_pass_at_k():
    """score_responses should compute pass@k per-prompt, mirroring IFEvalStrategy."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]],
            "prompt": ["Q1", "Q2"],
            "reference_answer": ["42", "100"],
        })
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kw: {"input_ids": [[0] * 10] * len(texts)}
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=2)
        results = [
            {"prompt": "Q1", "responses": ["Step 1: ...\n#### 42"]},      # correct
            {"prompt": "Q2", "responses": ["Step 1: ...\n#### 99"]},      # incorrect
        ]
        scores = strategy.score_responses(results, tokenizer)
        assert scores["pass_at_1"] == 0.5
        assert scores["num_prompts_evaluated"] == 2


def test_gsm8k_score_responses_flexible_fallback():
    """score_responses should use flexible extraction when #### is missing."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Q1"}]],
            "prompt": ["Q1"],
            "reference_answer": ["42"],
        })
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kw: {"input_ids": [[0] * 10] * len(texts)}
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        results = [
            {"prompt": "Q1", "responses": ["The answer is 42"]},  # no ####, flexible kicks in
        ]
        scores = strategy.score_responses(results, tokenizer)
        assert scores["pass_at_1"] == 1.0


def test_gsm8k_wandb_metrics():
    """wandb_metrics should prefix scores with eval/gsm8k_."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        scores = {"pass_at_1": 0.75, "num_prompts_evaluated": 10, "avg_response_length_tokens": 50.0}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/gsm8k_pass_at_1" in wandb_dict
        assert "eval/avg_response_length_tokens" in wandb_dict


def _get_function_params(filepath, func_name):
    """Parse a Python file's AST to extract function parameter names."""
    import ast
    from pathlib import Path
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [arg.arg for arg in node.args.args]
    raise ValueError(f"Function {func_name} not found in {filepath}")


def test_sft_training_accepts_primary_eval():
    """train_model_sft should accept primary_eval and monitor_evals, not ifeval_config."""
    params = _get_function_params("tuning/training/sft_training.py", "train_model_sft")
    assert "primary_eval" in params
    assert "monitor_evals" in params
    assert "ifeval_config" not in params


def test_dpo_training_accepts_primary_eval():
    """train_model_dpo should accept primary_eval and monitor_evals, not ifeval_config."""
    params = _get_function_params("tuning/training/dpo_training.py", "train_model_dpo")
    assert "primary_eval" in params
    assert "monitor_evals" in params
    assert "ifeval_config" not in params
