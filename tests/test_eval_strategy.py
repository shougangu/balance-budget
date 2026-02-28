# ABOUTME: Tests for eval strategy ABC and IFEval implementation.
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
        strategy = IFEvalStrategy(
            k_values=[1],
            n_samples=1,
            strict=True,
            num_prompts=1,
            tokenizer=MagicMock(),
        )
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
        strategy = IFEvalStrategy(
            k_values=[1, 5], n_samples=5, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
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
        strategy = IFEvalStrategy(
            k_values=[1], n_samples=1, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
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
        strategy = IFEvalStrategy(
            k_values=[1], n_samples=1, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
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
